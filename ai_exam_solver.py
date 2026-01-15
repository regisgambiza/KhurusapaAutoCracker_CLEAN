"""AI-Powered Exam Solver using Ollama llama3.1:8b - SMART BRUTE FORCE VERSION"""

import logging
import json
import re
import time
import os
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import requests

logger = logging.getLogger(__name__)


class OllamaAI:
    """Interface to Ollama AI for question answering."""
    
    def __init__(self, model: str = "llama3.1:8b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
    
    def ask(self, prompt: str, temperature: float = 0.3) -> str:
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "temperature": temperature
            }
            response = requests.post(self.api_url, json=payload, timeout=60)
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            return ""
    
    def answer_question(self, question: str, options: List[str]) -> Tuple[int, str]:
        options_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
        
        prompt = f"""You are taking an exam. Answer this multiple choice question by selecting the BEST answer.

Question: {question}

Options:
{options_text}

Respond ONLY with a JSON object in this exact format:
{{"answer": "A", "reasoning": "brief explanation"}}

Choose the letter (A, B, C, D, etc.) that corresponds to the best answer."""

        response = self.ask(prompt, temperature=0.2)
        
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                answer_letter = result.get("answer", "A").upper().strip()
                reasoning = result.get("reasoning", "No reasoning provided")
                answer_index = ord(answer_letter) - 65
                if 0 <= answer_index < len(options):
                    return answer_index, reasoning
        except Exception as e:
            logger.error(f"Failed to parse AI response: {e}")
        
        return 0, "Default to first option"


class SmartBruteForceSolver:
    """
    Smart brute force solver that:
    1. Uses AI to make initial educated guesses
    2. Tracks all attempts and results in JSON
    3. Cracks question-by-question, learning as it goes
    4. Locks down correct answers once found
    """
    
    def __init__(self, ai, questions: List[Dict], json_file: str = "exam_progress.json"):
        self.ai = ai
        self.questions = questions
        self.json_file = json_file
        self.state = self._load_state()
        
        # Initialize state if new
        if not self.state.get('initialized'):
            self._initialize_state()
    
    def _load_state(self) -> Dict:
        """Load state from JSON file."""
        if os.path.exists(self.json_file):
            try:
                with open(self.json_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                    logger.info(f"[JSON] Loaded existing state from {self.json_file}")
                    return state
            except Exception as e:
                logger.error(f"Failed to load state: {e}")
        
        return {}
    
    def _save_state(self):
        """Save state to JSON file."""
        try:
            self.state['last_updated'] = datetime.now().isoformat()
            with open(self.json_file, 'w', encoding='utf-8') as f:
                json.dump(self.state, indent=2, fp=f)
            logger.info(f"[JSON] State saved to {self.json_file}")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def _initialize_state(self):
        """Initialize fresh state for new exam."""
        logger.info("[INIT] Initializing new exam state...")
        
        self.state = {
            'initialized': True,
            'start_time': datetime.now().isoformat(),
            'total_questions': len(self.questions),
            'current_phase': 'initial_ai_attempt',
            'attempt_number': 0,
            'best_score': 0,
            'current_question_index': 0,  # Track which question we're cracking
            'questions': {},
            'attempt_history': []
        }
        
        # Initialize each question
        for question in self.questions:
            q_id = question['id']
            self.state['questions'][q_id] = {
                'question_text': question['question_text'],
                'options': [opt['text'] for opt in question['options']],
                'num_options': len(question['options']),
                'current_answer_index': None,
                'locked': False,
                'confirmed_correct_index': None,
                'tried_indices': [],
                'ai_reasoning': None,
                'status': 'untested'  # untested, testing, correct, incorrect
            }
        
        self._save_state()
    
    def get_initial_ai_answers(self) -> Dict[str, int]:
        """
        Phase 1: Use AI to reason and get initial best guesses.
        Returns dict of {question_id: option_index}
        """
        logger.info("\n" + "="*80)
        logger.info("[PHASE 1] AI Initial Reasoning")
        logger.info("="*80)
        
        answers = {}
        
        for question in self.questions:
            q_id = question['id']
            q_text = question['question_text']
            options = [opt['text'] for opt in question['options']]
            
            logger.info(f"\n{q_id}: {q_text[:50]}...")
            
            # Ask AI
            selected_idx, reasoning = self.ai.answer_question(q_text, options)
            
            # Store in state
            self.state['questions'][q_id]['current_answer_index'] = selected_idx
            self.state['questions'][q_id]['tried_indices'].append(selected_idx)
            self.state['questions'][q_id]['ai_reasoning'] = reasoning
            self.state['questions'][q_id]['status'] = 'testing'
            
            answers[q_id] = selected_idx
            
            logger.info(f"[AI] Selected option {selected_idx + 1}: {options[selected_idx][:50]}...")
            logger.info(f"[REASON] {reasoning}")
        
        self.state['current_phase'] = 'smart_brute_force'
        self._save_state()
        
        return answers
    
    def get_next_test_answers(self, last_score: int) -> Tuple[Dict[str, int], str, str]:
        """
        Get answers for the next test attempt.
        Returns: (answers_dict, current_q_id, status_message)
        """
        # Update best score
        if last_score > self.state['best_score']:
            self.state['best_score'] = last_score
        
        # Check if we've completed all questions
        all_locked = all(q['locked'] for q in self.state['questions'].values())
        if all_locked:
            return self._get_all_answers(), "", "ALL_QUESTIONS_LOCKED"
        
        # Get current question to test
        current_q_id = self._get_current_test_question()
        if not current_q_id:
            return self._get_all_answers(), "", "NO_MORE_QUESTIONS_TO_TEST"
        
        q_state = self.state['questions'][current_q_id]
        
        # If question is already locked, use confirmed answer
        if q_state['locked'] and q_state['confirmed_correct_index'] is not None:
            answers = self._get_all_answers()
            return answers, current_q_id, f"LOCKED_CORRECT_{current_q_id}"
        
        # Get next option to try for this question
        next_option_idx = self._get_next_option_to_try(current_q_id)
        
        if next_option_idx is None:
            # All options tried, mark as locked with no correct answer found
            q_state['locked'] = True
            q_state['status'] = 'exhausted'
            self._save_state()
            
            # Move to next question
            self.state['current_question_index'] = (self.state['current_question_index'] + 1) % len(self.questions)
            return self.get_next_test_answers(last_score)
        
        # Update state for this question
        q_state['current_answer_index'] = next_option_idx
        if next_option_idx not in q_state['tried_indices']:
            q_state['tried_indices'].append(next_option_idx)
        q_state['status'] = 'testing'
        
        # Build answer set with new option for testing question, locked answers for others
        answers = self._get_all_answers()
        answers[current_q_id] = next_option_idx
        
        logger.info(f"[CRACK] Testing {current_q_id} with option {next_option_idx + 1}")
        self._save_state()
        
        return answers, current_q_id, f"TESTING_{current_q_id}_OPT_{next_option_idx + 1}"
    
    def process_test_result(self, score: int, total: int, tested_q_id: str = None):
        """
        Process the result of a test attempt.
        """
        attempt_record = {
            'attempt_number': self.state['attempt_number'],
            'timestamp': datetime.now().isoformat(),
            'score': score,
            'total': total,
            'tested_question': tested_q_id,
            'answers': {q_id: q_state['current_answer_index'] 
                       for q_id, q_state in self.state['questions'].items()}
        }
        
        self.state['attempt_history'].append(attempt_record)
        self.state['attempt_number'] += 1
        
        # If we tested a specific question
        if tested_q_id and tested_q_id in self.state['questions']:
            q_state = self.state['questions'][tested_q_id]
            tested_option_idx = q_state['current_answer_index']
            prev_best_score = self.state['best_score']
            
            logger.info(f"[ANALYSIS] Score: {score}/{total}, Previous best: {prev_best_score}")
            
            if score > prev_best_score:
                # New option is CORRECT!
                logger.info(f"[CORRECT] {tested_q_id} option {tested_option_idx + 1} is CORRECT!")
                q_state['confirmed_correct_index'] = tested_option_idx
                q_state['locked'] = True
                q_state['status'] = 'correct'
                self.state['best_score'] = score
                
                # Move to next question
                self.state['current_question_index'] = (self.state['current_question_index'] + 1) % len(self.questions)
                
            elif score < prev_best_score:
                # New option is WRONG
                logger.info(f"[WRONG] {tested_q_id} option {tested_option_idx + 1} is wrong")
                q_state['status'] = 'incorrect'
                # Keep trying other options
                
            else:
                # Score unchanged - both options might be wrong
                logger.info(f"[NO_CHANGE] {tested_q_id}: Score unchanged, option {tested_option_idx + 1} might be wrong")
                q_state['status'] = 'incorrect'
        
        self._save_state()
    
    def _get_current_test_question(self) -> str:
        """Get the current question ID to test."""
        # Get question order
        question_ids = [q['id'] for q in self.questions]
        
        # Start from current question index
        start_idx = self.state['current_question_index']
        
        # Try to find next testable question
        for i in range(len(question_ids)):
            idx = (start_idx + i) % len(question_ids)
            q_id = question_ids[idx]
            q_state = self.state['questions'][q_id]
            
            # Skip locked questions
            if q_state['locked']:
                continue
                
            # Check if we have more options to try
            if len(q_state['tried_indices']) < q_state['num_options']:
                self.state['current_question_index'] = idx
                return q_id
        
        return None
    
    def _get_next_option_to_try(self, q_id: str) -> int:
        """Get the next option index to try for a question."""
        q_state = self.state['questions'][q_id]
        tried_indices = q_state['tried_indices']
        
        # Get all possible indices
        all_indices = list(range(q_state['num_options']))
        untried = [i for i in all_indices if i not in tried_indices]
        
        if not untried:
            return None
        
        # If this is the first time testing this question, use AI's suggestion
        if len(tried_indices) == 0:
            # Use the AI's initial answer
            return q_state['current_answer_index'] if q_state['current_answer_index'] is not None else untried[0]
        
        # For subsequent attempts, try untried options in order
        return untried[0]
    
    def _get_all_answers(self) -> Dict[str, int]:
        """Get current best answer for each question."""
        answers = {}
        for q_id, q_state in self.state['questions'].items():
            if q_state['locked'] and q_state['confirmed_correct_index'] is not None:
                # Use confirmed correct answer
                answers[q_id] = q_state['confirmed_correct_index']
            elif q_state['current_answer_index'] is not None:
                # Use current answer
                answers[q_id] = q_state['current_answer_index']
            else:
                # Default to first option
                answers[q_id] = 0
        
        return answers
    
    def get_status_report(self) -> str:
        """Generate a status report."""
        locked_count = sum(1 for q in self.state['questions'].values() 
                          if q['locked'] and q['confirmed_correct_index'] is not None)
        
        report_lines = [
            "\n" + "="*80,
            "SMART BRUTE FORCE SOLVER STATUS",
            "="*80,
            f"Phase: {self.state['current_phase']}",
            f"Attempt: {self.state['attempt_number']}",
            f"Best Score: {self.state['best_score']}/{self.state['total_questions']}",
            f"Confirmed Correct: {locked_count}/{self.state['total_questions']}",
            f"Current Test Question: Q{self.state['current_question_index'] + 1}",
            "-"*80
        ]
        
        for q_id, q_state in self.state['questions'].items():
            status = q_state['status']
            if q_state['locked']:
                status = "LOCKED✓" if q_state['confirmed_correct_index'] is not None else "LOCKED✗"
            
            report_lines.append(f"{q_id}: {status} | Tried: {len(q_state['tried_indices'])}/{q_state['num_options']} | "
                              f"Current: {q_state['current_answer_index']}")
        
        report_lines.append("="*80)
        
        return "\n".join(report_lines)


class QuestionExtractor:
    @staticmethod
    def extract_questions(page, num_questions: int = 10) -> List[Dict]:
        questions = []

        # 1️⃣ WAIT ONCE – inputs exist, nothing else
        page.wait_for_selector("input[type='radio'], input[type='checkbox']", timeout=10000)

        # 2️⃣ TAKE A SNAPSHOT (no more waiting after this)
        all_inputs = page.locator("input[type='radio'], input[type='checkbox']")
        all_labels = page.locator("label")

        total_inputs = all_inputs.count()
        total_labels = all_labels.count()

        logger.info(f"[DEBUG] Found {total_inputs} inputs and {total_labels} labels")

        if total_inputs == 0:
            logger.error("[!] No inputs found")
            return []

        options_per_question = total_inputs // num_questions

        # 3️⃣ EXTRACT FAST — no waits, no DOM walking
        for q_idx in range(num_questions):
            start = q_idx * options_per_question
            end = start + options_per_question

            options = []
            for i in range(start, min(end, total_inputs)):
                label_text = ""
                if i < total_labels:
                    # text_content() is now instant because we waited already
                    label_text = all_labels.nth(i).text_content().strip()

                options.append({
                    "text": label_text or f"Option {i-start+1}"
                })

            questions.append({
                "id": f"q{q_idx + 1}",
                "question_text": f"Question {q_idx + 1}",  # FAST & SAFE
                "options": options
            })

            logger.info(f"[OK] Extracted Q{q_idx + 1} ({len(options)} options)")

        return questions
    
    @staticmethod
    def _extract_question_near_input(input_elem, page) -> str:
        try:
            for selector in [
                'xpath=ancestor::div[contains(@class, "question")]',
                'xpath=ancestor::fieldset',
                'xpath=ancestor::div[contains(@class, "form")]',
                'xpath=preceding::*[self::h1 or self::h2 or self::h3 or self::h4 or self::strong][1]'
            ]:
                try:
                    parent = input_elem.locator(selector).first
                    text = parent.text_content().strip()
                    lines = [l.strip() for l in text.split('\n') if len(l.strip()) > 10]
                    if lines:
                        return lines[0]
                except:
                    continue
        except:
            pass
        return ""


class AIExamSolver:
    def _click_all_answers_correct_by_index(self, question: Dict, target_text: str) -> bool:
        """
        Special handler for 'All the answers are correct' option.
        Finds the option index within the question's options list and clicks by index.
        """
        try:
            qid = question['id']
            options = question['options']
            
            logger.info(f"[SPECIAL] Handling 'All answers correct' for {qid}")
            
            # Find which index contains "All the answers are correct"
            target_index = None
            for idx, opt in enumerate(options):
                opt_text = opt['text'].strip().lower()
                if "all" in opt_text and "answer" in opt_text and "correct" in opt_text:
                    target_index = idx
                    logger.info(f"  Found 'All answers correct' at index {idx}: '{opt['text']}'")
                    break
            
            if target_index is None:
                logger.error(f"  Could not find 'All answers correct' in options!")
                return False
            
            # Now click using the input element at that index
            # Get all inputs on page
            all_inputs = self.page.locator("input[type='radio'], input[type='checkbox']")
            total_inputs = all_inputs.count()
            
            # Calculate which questions we've seen
            q_num = int(qid.replace('q', '')) - 1  # q1 -> 0, q2 -> 1, etc.
            options_per_question = len(options)
            
            # Calculate global index of this specific option
            global_input_index = (q_num * options_per_question) + target_index
            
            logger.info(f"  Question {qid} (q_num={q_num}), target_index={target_index}")
            logger.info(f"  Global input index: {global_input_index} (total inputs: {total_inputs})")
            
            if global_input_index >= total_inputs:
                logger.error(f"  Calculated index {global_input_index} exceeds total inputs {total_inputs}")
                return False
            
            # Click the input directly by index
            try:
                target_input = all_inputs.nth(global_input_index)
                
                # Try clicking the input directly
                try:
                    target_input.click(force=True, timeout=5000)
                    logger.info(f"✓ Clicked input directly at index {global_input_index}")
                    return True
                except:
                    pass
                
                # If direct click fails, try clicking the associated label
                try:
                    input_id = target_input.get_attribute('id')
                    if input_id:
                        label = self.page.locator(f"label[for='{input_id}']")
                        if label.count() > 0:
                            label.first.click(force=True, timeout=5000)
                            logger.info(f"✓ Clicked label for input at index {global_input_index}")
                            return True
                except:
                    pass
                
                # Last resort: click the label at the same index
                all_labels = self.page.locator("label")
                if global_input_index < all_labels.count():
                    all_labels.nth(global_input_index).click(force=True, timeout=5000)
                    logger.info(f"✓ Clicked label at index {global_input_index}")
                    return True
                    
            except Exception as e:
                logger.error(f"  Failed to click by index: {e}")
                return False
                
        except Exception as e:
            logger.error(f"Error in _click_all_answers_correct_by_index: {e}")
            return False
    
    def retest_and_extract(self, num_questions: int = 10) -> bool:
        """Retest and re-extract questions with better error handling."""
        try:
            logger.info("Retesting → re-extracting all questions...")
            
            # Try different retest button patterns
            retest_patterns = [
                "button:has-text('Retest')",
                "button:has-text('Try Again')",
                "button:has-text('Restart')",
                "button:has-text('Redo')",
                "button:has-text('Again')",
                "a:has-text('Retest')"
            ]
            
            for pattern in retest_patterns:
                try:
                    retest_btn = self.page.locator(pattern).first
                    if retest_btn.count() > 0:
                        retest_btn.click()
                        time.sleep(2)
                        break
                except:
                    continue
            
            self.page.wait_for_load_state("networkidle", timeout=15000)
            time.sleep(1)
            
            # Re-extract questions
            return self.analyze_exam(num_questions)
            
        except Exception as e:
            logger.error(f"Retest failed: {e}")
            return False


    def __init__(self, page, ai: OllamaAI):
        self.page = page
        self.ai = ai
        self.questions: List[Dict] = []
        self.smart_solver: Optional[SmartBruteForceSolver] = None
        self.answers: Dict[str, Dict] = {}

    def analyze_exam(self, num_questions: int = 10) -> bool:
        logger.info(f"[*] Extracting questions (expecting {num_questions})...")
        self.questions = QuestionExtractor.extract_questions(self.page, num_questions)
        if not self.questions:
            logger.error("Failed to extract any questions!")
            return False
        
        logger.info(f"[OK] Extracted {len(self.questions)} questions")
        
        # Initialize smart solver
        self.smart_solver = SmartBruteForceSolver(self.ai, self.questions)
        return True

    def get_answers_for_attempt(self, is_first_attempt: bool = False) -> bool:
        """Get answers from smart solver for current attempt."""
        if not self.smart_solver:
            logger.error("Smart solver not initialized!")
            return False
        
        if is_first_attempt:
            # Phase 1: Get initial AI answers
            logger.info("\n[PHASE 1] Getting initial AI answers...")
            answers_dict = self.smart_solver.get_initial_ai_answers()
        else:
            # Phase 2: Smart brute force
            logger.info("\n[PHASE 2] Smart brute force - getting next test answers...")
            answers_dict, tested_q_id, status = self.smart_solver.get_next_test_answers(
                self.smart_solver.state['best_score']
            )
            
            if status == "ALL_QUESTIONS_LOCKED":
                logger.info("[SUCCESS] All questions have been locked with correct answers!")
                return False
            elif status == "NO_MORE_QUESTIONS_TO_TEST":
                logger.info("[WARNING] No more questions to test!")
                return False
        
        # Convert answers dict to our format
        self.answers = {}
        for q_id, answer_idx in answers_dict.items():
            # Find the question
            question = next((q for q in self.questions if q['id'] == q_id), None)
            if question and 0 <= answer_idx < len(question['options']):
                option_text = question['options'][answer_idx]['text']
                self.answers[q_id] = {
                    'option_text': option_text,
                    'answer_index': answer_idx
                }
        
        logger.info(f"[OK] Prepared answers for {len(self.answers)} questions")
        logger.info(self.smart_solver.get_status_report())
        return True

    def click_answers(self) -> bool:
        """Enhanced version with special handling for 'All the answers are correct'"""
        logger.info("\n[*] Clicking answers...")
        success_count = 0

        for q in self.questions:
            qid = q['id']
            if qid not in self.answers:
                logger.warning(f"No answer for {qid}")
                continue

            target_text = self.answers[qid]['option_text'].strip()
            target_text = ' '.join(target_text.split())  # normalize spaces
            question_text = q['question_text']

            logger.info(f"  Trying to click for {qid}: '{target_text[:70]}...'")

            # Special handling for "All the answers are correct"
            if "all" in target_text.lower() and "answer" in target_text.lower() and "correct" in target_text.lower():
                if self._click_all_answers_correct_by_index(q, target_text):
                    success_count += 1
                    time.sleep(1.0)
                    continue
                else:
                    logger.error(f"Failed to click 'All answers correct' for {qid}")
                    # Fall back to regular clicking
                    if self._click_by_text(target_text):
                        success_count += 1
                        time.sleep(1.0)
                    continue

            # Regular clicking for other answers
            if self._click_by_text(target_text):
                success_count += 1
                time.sleep(1.0)
            else:
                logger.error(f"Failed to click answer for {qid}: {target_text[:60]}...")

        logger.info(f"Successfully clicked {success_count}/{len(self.questions)} answers")
        return success_count == len(self.questions)

    def _click_by_text(self, target_text: str) -> bool:
        
        """Simplified click by text method."""
        try:
            # Try exact match
            labels = self.page.locator("label").filter(has_text=re.compile(re.escape(target_text), re.IGNORECASE))
            if labels.count() > 0:
                try:
                    labels.first.click(force=True, timeout=5000)
                    logger.debug(f"✓ Clicked using exact match")
                    return True
                except:
                    pass
            
            # Try partial match for longer texts
            if len(target_text) > 20:
                partial = target_text[:30]
                labels = self.page.locator("label").filter(has_text=re.compile(re.escape(partial), re.IGNORECASE))
                if labels.count() > 0:
                    try:
                        labels.first.click(force=True, timeout=5000)
                        logger.debug(f"✓ Clicked using partial match")
                        return True
                    except:
                        pass
            
            logger.error(f"✗ Failed to click: '{target_text}'")
            return False
            
        except Exception as e:
            logger.error(f"Click error: {e}")
            return False

    def submit_and_check(self) -> Tuple[int, int, bool]:
        """Submit answers and check score."""
        try:
            logger.info("Submitting...")

            submit_btn = self.page.get_by_role(
                "button",
                name=re.compile(r"submit|finish|send|complete", re.I)
            )
            submit_btn.click()
            time.sleep(3.5)

            # Find score text
            score_locator = self.page.locator(
                "text=/\\d+\\s*\\/\\s*\\d+/"
            ).filter(
                has_text=re.compile(r"score|result|mark|grade", re.I)
            )

            score_text = ""
            for i in range(score_locator.count()):
                text = score_locator.nth(i).inner_text().strip()
                if re.search(r"\d+\s*/\s*\d+", text):
                    score_text = text
                    break

            if not score_text:
                raise Exception("Score text not found")

            logger.info(f"Result text: {score_text}")

            match = re.search(r'(\d+)\s*/\s*(\d+)', score_text)
            if match:
                score = int(match.group(1))
                total = int(match.group(2))
                return score, total, score == total

        except Exception as e:
            logger.error(f"Failed to submit or read score: {e}")

        return 0, 0, False

    def retest(self, num_questions: int = 10) -> bool:
        """Reset the exam for next attempt."""
        try:
            logger.info("Retesting → resetting exam...")
            retest_btn = self.page.get_by_role("button", name=re.compile(r"retest|try again|redo|restart|again", re.I))
            retest_btn.click()
            time.sleep(2.5)

            self.page.wait_for_load_state("networkidle", timeout=15000)
            
            # Re-extract questions (order might change)
            self.questions = QuestionExtractor.extract_questions(self.page, num_questions)
            if not self.questions:
                logger.error("Failed to re-extract questions!")
                return False
            
            # Update smart solver with new questions (same IDs but fresh extraction)
            if self.smart_solver:
                self.smart_solver.questions = self.questions
                # Update question texts in state
                for q in self.questions:
                    q_id = q['id']
                    if q_id in self.smart_solver.state['questions']:
                        self.smart_solver.state['questions'][q_id]['question_text'] = q['question_text']
                        self.smart_solver.state['questions'][q_id]['options'] = [opt['text'] for opt in q['options']]
                
                self.smart_solver._save_state()
            
            return True
            
        except Exception as e:
            logger.error(f"Retest failed: {e}")
            return False

    def process_attempt_result(self, score: int, total: int, is_first_attempt: bool = False):
        """Process the result of an attempt."""
        if self.smart_solver:
            if is_first_attempt:
                # For first attempt, we tested all questions
                self.smart_solver.process_test_result(score, total)
            else:
                # For subsequent attempts, we need to determine which question was tested
                # The smart solver tracks this internally
                self.smart_solver.process_test_result(score, total)
            
            logger.info(self.smart_solver.get_status_report())


def run_ai_solver(page, num_questions: int = 10, max_attempts: int = 30):
    ai = OllamaAI()
    solver = AIExamSolver(page, ai)
    
    # First, extract questions
    if not solver.analyze_exam(num_questions):
        logger.error("Failed to extract questions!")
        return
    
    attempt = 0
    last_score = 0
    
    while attempt < max_attempts:
        attempt += 1
        logger.info(f"\n{'='*80}")
        logger.info(f"ATTEMPT {attempt}/{max_attempts}")
        logger.info(f"{'='*80}")
        
        # Get answers for this attempt
        is_first_attempt = (attempt == 1)
        if not solver.get_answers_for_attempt(is_first_attempt):
            logger.warning("Failed to get answers for this attempt!")
            break
        
        # Click answers and submit
        logger.info("\nClicking answers...")
        if not solver.click_answers():
            logger.warning("Some answers could not be clicked!")
        
        # Submit and get score
        score, total, is_perfect = solver.submit_and_check()
        logger.info(f"Score: {score}/{total}")
        
        # Process attempt result
        solver.process_attempt_result(score, total, is_first_attempt)
        
        if is_perfect:
            logger.info("🎉 PERFECT SCORE ACHIEVED!")
            break
        
        if attempt >= max_attempts:
            logger.info("Max attempts reached.")
            break
        
        # Retest for next attempt
        logger.info("\nRetesting for next attempt...")
        if not solver.retest(num_questions):
            logger.error("Retest failed!")
            break
    
    logger.info("\n" + "="*80)
    logger.info("EXAM SOLVING COMPLETED")
    logger.info("="*80)
    logger.info(f"Final score: {last_score}/{num_questions}")
    logger.info(f"Total attempts: {attempt}")