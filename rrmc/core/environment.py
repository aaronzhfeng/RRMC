"""
AR-Bench Environment Wrapper for RRMC.

Provides a unified interface over AR-Bench tasks (DC, SP, GN).
"""

import json
import re
from typing import Dict, List, Optional, Any, Tuple
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum

from .llm import LLMWrapper


class TaskType(Enum):
    """AR-Bench task types."""
    DC = "detective_cases"
    SP = "situation_puzzles"
    GN = "guessing_numbers"


@dataclass
class ActionASK:
    """Ask action for DC/SP or guess for GN."""
    question: str
    suspect: Optional[str] = None  # For DC: which suspect to ask


@dataclass
class ActionANSWER:
    """Final answer action."""
    answer: str  # DC: suspect name/index, SP: explanation, GN: 4-digit number


@dataclass
class StepResult:
    """Result from environment step."""
    observation: str  # NPC response or feedback
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)


class ARBenchEnv:
    """
    Unified environment wrapper for AR-Bench tasks.

    Supports:
    - DC (Detective Cases): Interrogate suspects to identify murderer
    - SP (Situation Puzzles): Ask yes/no questions to explain hidden story
    - GN (Guessing Numbers): Bulls-and-cows style number guessing
    """

    # DC response template
    DC_RESPOND_TEMPLATE = """You will now play the role of a suspect with a specific task and answer questions based on the stories provided below. When responding to a question, your answers should be limited in one sentence.

Your Name: {name}
Your Task: {task}
Your Story: {story}
"""

    def __init__(
        self,
        task_type: TaskType,
        data_path: str,
        llm: Optional[LLMWrapper] = None,
        response_temperature: float = 0.7,
        response_top_p: float = 0.7,
        max_turns: int = 25,
    ):
        """
        Initialize AR-Bench environment.

        Args:
            task_type: Type of task (DC, SP, GN)
            data_path: Path to dataset JSON file
            llm: LLM wrapper for NPC responses (required for DC/SP)
            response_temperature: Temperature for NPC responses
            response_top_p: Top-p for NPC responses
            max_turns: Maximum number of turns per episode
        """
        self.task_type = task_type
        self.llm = llm
        self.response_temperature = response_temperature
        self.response_top_p = response_top_p

        # Load dataset
        with open(data_path, 'r') as f:
            self.dataset = json.load(f)

        # Current state
        self.current_puzzle_idx: Optional[int] = None
        self.current_puzzle: Optional[Dict] = None
        self.history: List[Dict[str, str]] = []
        self.turn: int = 0
        self.max_turns: int = max_turns
        self.done: bool = False

        # DC-specific: response agents for suspects
        self._suspect_agents: Dict[str, List[Dict]] = {}

        # Cache initial observation (to avoid resetting in get_state)
        self._initial_obs: Dict[str, Any] = {}

    def __len__(self) -> int:
        """Number of puzzles in dataset."""
        return len(self.dataset)

    def reset(self, puzzle_idx: int) -> Dict[str, Any]:
        """
        Reset environment with a specific puzzle.

        Args:
            puzzle_idx: Index into dataset

        Returns:
            Initial observation dict with case info
        """
        self.current_puzzle_idx = puzzle_idx
        self.current_puzzle = self.dataset[puzzle_idx]
        self.history = []
        self.turn = 0
        self.done = False

        if self.task_type == TaskType.DC:
            obs = self._reset_dc()
        elif self.task_type == TaskType.SP:
            obs = self._reset_sp()
        elif self.task_type == TaskType.GN:
            obs = self._reset_gn()
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")

        # Cache initial observation for get_state()
        self._initial_obs = obs
        return obs

    def _reset_dc(self) -> Dict[str, Any]:
        """Reset for Detective Cases task."""
        puzzle = self.current_puzzle
        init_info = puzzle["initial_information"]

        # Initialize suspect response agents
        self._suspect_agents = {}
        for suspect in puzzle["suspects"]:
            system_prompt = self.DC_RESPOND_TEMPLATE.format(
                name=suspect["name"],
                task=suspect["task"],
                story=suspect["story"],
            )
            self._suspect_agents[suspect["name"]] = [
                {"role": "system", "content": system_prompt}
            ]

        # Format initial information
        init_info_str = self._format_dc_initial_info(init_info)

        # Get suspect names and choices
        suspect_names = [s["name"] for s in init_info["suspect"]]
        choices = {
            chr(ord('A') + i): name
            for i, name in enumerate(suspect_names)
        }

        return {
            "task_type": "DC",
            "initial_info": init_info_str,
            "suspect_names": suspect_names,
            "choices": choices,
            "ground_truth": puzzle["label"],  # Index of true murderer
            "ground_truth_name": suspect_names[puzzle["label"]],
        }

    def _format_dc_initial_info(self, init_info: Dict) -> str:
        """Format DC initial information as string."""
        victim = init_info["victim"]
        suspect_strs = []
        for i, s in enumerate(init_info["suspect"]):
            suspect_strs.append(
                f"- suspect {i+1}:\n"
                f"  - name: {s['name']}\n"
                f"  - introduction: {s['introduction']}"
            )

        return f"""time: {init_info['time']}
location: {init_info['location']}
victim:
- name: {victim['name']}
- introduction: {victim['introduction']}
- cause of death: {victim['cause_of_death']}
- murder weapon: {victim['murder_weapon']}
The investigation focuses on five suspects, one of whom is the true murderer:
{chr(10).join(suspect_strs)}
"""

    def _reset_sp(self) -> Dict[str, Any]:
        """Reset for Situation Puzzles task."""
        puzzle = self.current_puzzle
        # SP structure varies - adapt based on actual format
        return {
            "task_type": "SP",
            "surface": puzzle.get("surface", puzzle.get("initial_information", "")),
            "ground_truth": puzzle.get("explanation", puzzle.get("bottom", "")),
        }

    def _reset_gn(self) -> Dict[str, Any]:
        """Reset for Guessing Numbers task."""
        puzzle = self.current_puzzle
        # GN data can be just a string (the secret) or a dict
        if isinstance(puzzle, str):
            secret = puzzle
        else:
            secret = puzzle.get("secret", puzzle.get("answer", str(puzzle)))
        return {
            "task_type": "GN",
            "rules": "Guess a 4-digit number with unique digits (0-9, each digit can only appear once). Feedback: digits in correct position and digits in wrong position.",
            "ground_truth": secret,
        }

    def step(self, action: Any) -> StepResult:
        """
        Take a step in the environment.

        Args:
            action: ActionASK or ActionANSWER

        Returns:
            StepResult with observation, done flag, and info
        """
        if self.done:
            return StepResult(
                observation="Episode already done.",
                done=True,
                info={"error": "Episode ended"}
            )

        self.turn += 1

        if isinstance(action, ActionANSWER):
            return self._handle_answer(action)
        elif isinstance(action, ActionASK):
            return self._handle_ask(action)
        else:
            # Support dict-based actions for flexibility
            if isinstance(action, dict):
                if action.get("type") == "ANSWER":
                    return self._handle_answer(ActionANSWER(answer=action["answer"]))
                elif action.get("type") == "ASK":
                    return self._handle_ask(ActionASK(
                        question=action["question"],
                        suspect=action.get("suspect")
                    ))
            raise ValueError(f"Invalid action type: {type(action)}")

    def _handle_answer(self, action: ActionANSWER) -> StepResult:
        """Handle final answer submission."""
        self.done = True

        if self.task_type == TaskType.DC:
            return self._evaluate_dc_answer(action.answer)
        elif self.task_type == TaskType.SP:
            return self._evaluate_sp_answer(action.answer)
        elif self.task_type == TaskType.GN:
            return self._evaluate_gn_answer(action.answer)
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")

    def _evaluate_dc_answer(self, answer: str) -> StepResult:
        """Evaluate DC answer."""
        puzzle = self.current_puzzle
        ground_truth_idx = puzzle["label"]
        suspect_names = [s["name"] for s in puzzle["initial_information"]["suspect"]]
        ground_truth_name = suspect_names[ground_truth_idx]

        # Parse answer - could be letter (A-E), index (0-4), or name
        pred_idx = self._parse_dc_answer(answer, suspect_names)

        correct = pred_idx == ground_truth_idx

        return StepResult(
            observation=f"Your answer: {answer}",
            done=True,
            info={
                "correct": correct,
                "prediction": pred_idx,
                "ground_truth": ground_truth_idx,
                "ground_truth_name": ground_truth_name,
                "turns_used": self.turn,
            }
        )

    def _parse_dc_answer(self, answer: str, suspect_names: List[str]) -> int:
        """Parse DC answer to suspect index."""
        answer = answer.strip()

        # Check for letter A-E
        letter_match = re.search(r'\b([A-E])\b', answer.upper())
        if letter_match:
            return ord(letter_match.group(1)) - ord('A')

        # Check for digit 1-5 (suspect numbering is often 1-based)
        one_based_match = re.search(r'\b([1-5])\b', answer)
        if one_based_match:
            return int(one_based_match.group(1)) - 1

        # Check for digit 0-4 (0-based indices)
        zero_based_match = re.search(r'\b([0-4])\b', answer)
        if zero_based_match:
            return int(zero_based_match.group(1))

        # Check for name match
        answer_lower = answer.lower()
        for i, name in enumerate(suspect_names):
            if name.lower() in answer_lower or answer_lower in name.lower():
                return i

        # Check for ordinal/number words (e.g., "first suspect", "suspect two")
        word_map = {
            "first": 0, "1st": 0, "one": 0,
            "second": 1, "2nd": 1, "two": 1,
            "third": 2, "3rd": 2, "three": 2,
            "fourth": 3, "4th": 3, "four": 3,
            "fifth": 4, "5th": 4, "five": 4,
        }
        for word, idx in word_map.items():
            if re.search(rf'\b{re.escape(word)}\b', answer_lower):
                return idx

        # Default: try to extract any number
        return 0

    @staticmethod
    def _bag_f1(prediction: str, ground_truth: str, level: str = "char") -> float:
        """
        Bag-of-items F1 score matching AR-Bench's scoring semantics.

        Args:
            prediction: Predicted text
            ground_truth: Ground truth text
            level: "char" for character-level, "word" for word-level
        """
        if level == "word":
            pred_items = prediction.lower().split()
            gt_items = ground_truth.lower().split()
        else:
            pred_items = list(prediction.lower())
            gt_items = list(ground_truth.lower())
        if not pred_items or not gt_items:
            return 0.0
        common = Counter(pred_items) & Counter(gt_items)
        num_same = sum(common.values())
        if num_same == 0:
            return 0.0
        precision = num_same / len(pred_items)
        recall = num_same / len(gt_items)
        return (2 * precision * recall) / (precision + recall)

    def _evaluate_sp_answer(self, answer: str) -> StepResult:
        """Evaluate SP answer using bag-of-characters F1 (AR-Bench compatible)."""
        ground_truth = self.current_puzzle.get("explanation",
                        self.current_puzzle.get("bottom", ""))

        f1_char = self._bag_f1(answer, ground_truth, level="char")
        f1_word = self._bag_f1(answer, ground_truth, level="word")

        # SP uses F1 threshold for correctness (typically F1 > 0.5 is considered correct)
        correct = f1_char > 0.5

        return StepResult(
            observation=f"Your explanation submitted.",
            done=True,
            info={
                "correct": correct,
                "prediction": answer,
                "ground_truth": ground_truth,
                "f1_score": f1_char,
                "f1_score_char": f1_char,
                "f1_score_word": f1_word,
                "turns_used": self.turn,
            }
        )

    def _evaluate_gn_answer(self, answer: str) -> StepResult:
        """Evaluate GN answer."""
        puzzle = self.current_puzzle
        if isinstance(puzzle, str):
            secret = puzzle
        else:
            secret = puzzle.get("secret", puzzle.get("answer", ""))

        # Parse and validate guess
        guess = re.sub(r'[^0-9]', '', answer)[:4]

        correct = guess == secret

        return StepResult(
            observation=f"Your guess: {guess}",
            done=True,
            info={
                "correct": correct,
                "prediction": guess,
                "ground_truth": secret,
                "guess": guess,      # Keep for backwards compatibility
                "secret": secret,    # Keep for backwards compatibility
                "turns_used": self.turn,
            }
        )

    def _handle_ask(self, action: ActionASK) -> StepResult:
        """Handle asking a question."""
        if self.turn > self.max_turns:
            self.done = True
            return StepResult(
                observation="Maximum turns reached. Please provide final answer.",
                done=True,
                info={"forced_end": True}
            )

        if self.task_type == TaskType.DC:
            return self._ask_dc(action)
        elif self.task_type == TaskType.SP:
            return self._ask_sp(action)
        elif self.task_type == TaskType.GN:
            return self._ask_gn(action)
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")

    def _ask_dc(self, action: ActionASK) -> StepResult:
        """Ask a question to a DC suspect."""
        if not self.llm:
            raise ValueError("LLM wrapper required for DC task")

        suspect = action.suspect
        question = action.question

        # Validate suspect name
        if suspect not in self._suspect_agents:
            # Try fuzzy match
            for name in self._suspect_agents:
                if suspect and (suspect.lower() in name.lower() or name.lower() in suspect.lower()):
                    suspect = name
                    break
            else:
                suspect = list(self._suspect_agents.keys())[0]

        # Add question to suspect's conversation
        self._suspect_agents[suspect].append({"role": "user", "content": question})

        # Get response
        response = self.llm.generate(
            messages=self._suspect_agents[suspect],
            temperature=self.response_temperature,
            top_p=self.response_top_p,
            max_tokens=256,
        )

        feedback = response.content

        # Add response to conversation history
        self._suspect_agents[suspect].append({"role": "assistant", "content": feedback})

        # Record in history
        self.history.append({
            "turn": self.turn,
            "suspect": suspect,
            "question": question,
            "feedback": feedback,
        })

        return StepResult(
            observation=feedback,
            done=False,
            info={
                "suspect": suspect,
                "question": question,
                "turn": self.turn,
            }
        )

    # AR-Bench SP referee system prompt (2-shot, from arbench/reasoner/sp/prompt.py)
    SP_REFEREE_TEMPLATE = """You are the referee of a game where players are shown a <Surface> and you are given the <Bottom>. You need to understand the entire story based on both the <Surface> and <Bottom>. Players will ask questions based on the <Surface>, and you need to judge whether their guesses are correct. Please strictly adhere to answering with only three specified responses: Yes, No, or Unknown, without any explanation.

## Judging Rules
- If the player's question matches the given <Surface> and <Bottom>: Please only answer "Yes" without any explanation.
- If the player's question contradicts the given story: Please only answer "No" without any explanation.
- If the answer to the player's question cannot be found in the <Surface> and <Bottom>, and cannot be deduced through reasoning: Please only answer "Unknown" without any explanation.

## Important Notes
1. Fully understand the cause, process, and outcome of the entire story, and make logical inferences.
2. If a conclusion cannot be drawn from the provided story or through reasonable inference, answer "Unknown".
3. Strictly adhere to answering with only the three specified responses: Yes, No, or Unknown. Do not provide any additional explanations.

## Examples

### Example 1: The Hiccuping Man
<Surface>
A man walks into a bar and asks the bartender for a glass of water. The bartender suddenly pulls out a gun and points it at him. The man smiles and says, "Thank you!" then calmly leaves. What happened?

<Bottom>
The man had hiccups and wanted a glass of water to cure them. The bartender realized this and chose to scare him with a gun. The man's hiccups disappeared due to the sudden shock, so he sincerely thanked the bartender before leaving.

Possible questions and corresponding answers:
Q: Does the man have a chronic illness? A: Unknown
Q: Was the man scared away? A: No
Q: Did the bartender want to kill the man? A: No
Q: Did the bartender intend to scare the man? A: Yes
Q: Did the man sincerely thank the bartender? A: Yes

### Example 2: The Four-Year-Old Mother
<Surface>
A five-year-old kindergartener surprisingly claims that her mother is only four years old. Puzzled, I proposed a home visit. When I arrived at her house, I saw a horrifying scene...

<Bottom>
I saw several women chained up in her house, with a fierce-looking, ugly brute standing nearby. The kindergartener suddenly displayed an eerie smile uncharacteristic of her age... It turns out she's actually 25 years old but suffers from a condition that prevents her from growing. The brute is her brother, and she lures kindergarten teachers like us to their house to help her brother find women... Her "four-year-old mother" is actually a woman who was tricked and has been held captive for four years...

Possible questions and corresponding answers:
Q: Is the child already dead? A: No
Q: Is the child actually an adult? A: Yes
Q: Does the child have schizophrenia? A: Unknown
Q: Am I in danger? A: Yes

## Question Content
### Surface
{surface}

### Bottom
{bottom}

Now, please judge the following player questions:
"""

    def _ask_sp(self, action: ActionASK) -> StepResult:
        """Ask a yes/no question for SP using AR-Bench's referee prompt."""
        if not self.llm:
            raise ValueError("LLM wrapper required for SP task")

        question = action.question
        surface = self.current_puzzle.get("surface",
                   self.current_puzzle.get("initial_information", ""))
        bottom = self.current_puzzle.get("explanation",
                  self.current_puzzle.get("bottom", ""))

        system_prompt = self.SP_REFEREE_TEMPLATE.format(
            surface=surface, bottom=bottom
        )

        # Build conversation: system prompt + previous Q&A + new question
        messages = [{"role": "system", "content": system_prompt}]
        for h in self.history:
            messages.append({"role": "user", "content": h["question"]})
            messages.append({"role": "assistant", "content": h["feedback"]})
        messages.append({"role": "user", "content": question})

        response = self.llm.generate(
            messages=messages,
            temperature=0.3,
            max_tokens=32,
        )

        # Normalize to Yes/No/Unknown (AR-Bench labels)
        raw = response.content.strip()
        raw_upper = raw.upper()
        if raw_upper.startswith("YES") or raw_upper == "YES":
            feedback = "Yes"
        elif raw_upper.startswith("NO") or raw_upper == "NO":
            feedback = "No"
        else:
            feedback = "Unknown"

        self.history.append({
            "turn": self.turn,
            "question": question,
            "feedback": feedback,
        })

        return StepResult(
            observation=feedback,
            done=False,
            info={"question": question, "turn": self.turn}
        )

    def _ask_gn(self, action: ActionASK) -> StepResult:
        """Make a guess for GN (symbolic feedback)."""
        guess = re.sub(r'[^0-9]', '', action.question)[:4]
        puzzle = self.current_puzzle
        if isinstance(puzzle, str):
            secret = puzzle
        else:
            secret = puzzle.get("secret", puzzle.get("answer", ""))

        # Validate guess has 4 unique digits
        if len(guess) != 4 or len(set(guess)) != 4:
            return StepResult(
                observation="Invalid guess. Must be exactly 4 unique digits.",
                done=False,
                info={"valid": False}
            )

        # Calculate bulls and cows
        bulls = sum(g == s for g, s in zip(guess, secret))
        cows = sum(min(guess.count(d), secret.count(d)) for d in set(guess)) - bulls

        # Check if solved
        if bulls == 4:
            self.done = True
            # Use AR-Bench's feedback format
            observation = f"{bulls} digits are present in the answer and in the correct positions \n0 digits are present in the answer but in the different positions \n"
            return StepResult(
                observation=observation,
                done=True,
                info={
                    "bulls": bulls,
                    "cows": cows,
                    "correct": True,
                    "prediction": guess,
                    "ground_truth": secret,
                    "turns_used": self.turn,
                }
            )

        self.history.append({
            "turn": self.turn,
            "guess": guess,
            "bulls": bulls,
            "cows": cows,
        })

        # Use AR-Bench's feedback format (eval_prompt)
        observation = f"{bulls} digits are present in the answer and in the correct positions \n{cows} digits are present in the answer but in the different positions \n"
        return StepResult(
            observation=observation,
            done=False,
            info={"bulls": bulls, "cows": cows, "guess": guess}
        )

    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history."""
        return self.history.copy()

    def get_history_string(self) -> str:
        """Get history as formatted string."""
        if self.task_type == TaskType.DC:
            lines = []
            for h in self.history:
                lines.append(
                    f"Turn {h['turn']}: Question for {h['suspect']}: {h['question']} "
                    f"Feedback: {h['feedback']}"
                )
            return "\n".join(lines)
        elif self.task_type == TaskType.SP:
            lines = []
            for h in self.history:
                lines.append(f"Q: {h['question']} A: {h['feedback']}")
            return "\n".join(lines)
        elif self.task_type == TaskType.GN:
            lines = []
            for i, h in enumerate(self.history):
                lines.append(
                    f"Turn {i+1}: Guess: {h['guess']}, Feedback: {h['bulls']} digits are present in the answer and in the correct positions, "
                    f"{h['cows']} digits are present in the answer but in the different positions"
                )
            return "\n".join(lines)
        return ""

    def get_state(self) -> Dict[str, Any]:
        """Get current state for uncertainty estimation."""
        # Use cached initial observation (don't reset!)
        obs = getattr(self, "_initial_obs", {}) or {}
        return {
            "task_type": self.task_type.value,
            "puzzle_idx": self.current_puzzle_idx,
            "turn": self.turn,
            "history": self.history.copy(),
            "history_string": self.get_history_string(),
            "done": self.done,
            **obs,
        }

    def get_ground_truth(self) -> Any:
        """
        Get the ground truth answer for the current puzzle.

        Returns:
            Ground truth in appropriate format for the task type:
            - DC: suspect index (0-4)
            - SP: explanation string
            - GN: secret number string
        """
        if self.current_puzzle is None:
            return None

        if self.task_type == TaskType.DC:
            return self.current_puzzle["label"]
        elif self.task_type == TaskType.SP:
            return self.current_puzzle.get("explanation",
                    self.current_puzzle.get("bottom", ""))
        elif self.task_type == TaskType.GN:
            puzzle = self.current_puzzle
            if isinstance(puzzle, str):
                return puzzle
            return puzzle.get("secret", puzzle.get("answer", ""))
        return None
