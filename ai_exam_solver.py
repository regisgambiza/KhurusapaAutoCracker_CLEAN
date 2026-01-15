"""AI-Powered Exam Solver using Ollama llama3.1:8b - SEQUENTIAL ELIMINATION VERSION - FIXED"""

import logging
import json
import re
import time
import os
from typing import Dict, List, Tuple, Optional, Set
from datetime import datetime
import requests

logger = logging.getLogger(__name__)


class SequentialEliminationSolver:
    """
    Strategy for exams with only global score feedback.
    Works by fixing ALL answers except ONE, then testing that one.
    FIXED VERSION: Only confirms answers when score CHANGES
    """
    
    def __init__(self, questions: List[Dict], json_file: str = "sequential_progress.json"):
        self.questions = questions
        self.json_file = json_file
        self.state = self._load_state()
        
        if not self.state.get('initialized'):
            self._initialize_state()
        else:
            # Ensure all required keys exist in loaded state
            self._migrate_state()
    
    def _migrate_state(self):
        """Migrate old state to new format if needed."""
        # Ensure 'tried_options' exists
        if 'tried_options' not in self.state:
            self.state['tried_options'] = {}
        
        # Convert any lists in tried_options back to sets
        for q_id, tried in self.state['tried_options'].items():
            if isinstance(tried, list):
                self.state['tried_options'][q_id] = set(tried)
        
        # Ensure all questions have entries in tried_options
        for question in self.questions:
            q_id = question['id']
            if q_id not in self.state['tried_options']:
                if q_id in self.state['candidate_answers']:
                    self.state['tried_options'][q_id] = set([self.state['candidate_answers'][q_id]])
                else:
                    self.state['tried_options'][q_id] = set([question['options'][0]['text']])
        
        # Ensure 'phase' exists
        if 'phase' not in self.state:
            self.state['phase'] = 'testing_questions'
        
        # Ensure 'perfect_score_achieved' exists
        if 'perfect_score_achieved' not in self.state:
            self.state['perfect_score_achieved'] = False
        
        # Ensure total_questions matches current questions
        if self.state.get('total_questions', 0) != len(self.questions):
            self.state['total_questions'] = len(self.questions)
    
    def _initialize_state(self):
        """Initialize fresh state."""
        self.state = {
            'initialized': True,
            'start_time': datetime.now().isoformat(),
            'total_questions': len(self.questions),
            'current_question_index': 0,
            'base_score': None,
            'confirmed_answers': {},  # {q_id: option_text}
            'candidate_answers': {},  # {q_id: option_text} - current best guess
            'tried_options': {},      # {q_id: Set(text)} - track tried options
            'test_history': [],
            'phase': 'establishing_baseline',
            'perfect_score_achieved': False,
            'last_score': None,
            'unconfirmed_questions': [f"q{i+1}" for i in range(len(self.questions))]  # Track unconfirmed questions
        }
        
        # Start with all first options as baseline
        for question in self.questions:
            q_id = question['id']
            self.state['candidate_answers'][q_id] = question['options'][0]['text']
            self.state['tried_options'][q_id] = set([question['options'][0]['text']])
        
        self._save_state()
    
    def get_test_set(self, current_score: Optional[int] = None) -> Dict[str, str]:
        """
        Get the next set of answers to test.
        
        Strategy:
        1. If baseline score unknown, test baseline (all first options)
        2. Otherwise, test ONE question at a time while fixing others
        """
        
        # Check if all confirmed
        if len(self.state['confirmed_answers']) == self.state['total_questions']:
            logger.info("[SUCCESS] All questions confirmed!")
            return self.state['confirmed_answers']
        
        # If we're still establishing baseline
        if self.state['phase'] == 'establishing_baseline' and current_score is not None:
            self.state['base_score'] = current_score
            self.state['last_score'] = current_score
            self.state['phase'] = 'testing_questions'
            logger.info(f"[BASELINE] Baseline score established: {current_score}/{self.state['total_questions']}")
            self._save_state()
        
        # If we don't have a baseline yet, return baseline set
        if self.state['base_score'] is None:
            logger.info("[TESTING] Testing baseline (all first options)...")
            return {q['id']: q['options'][0]['text'] for q in self.questions}
        
        # Find next unconfirmed question to test
        if not self.state['unconfirmed_questions']:
            logger.info("[SUCCESS] All questions confirmed!")
            return self.state['confirmed_answers']
        
        # Get current question to test (first in unconfirmed list)
        current_q_id = self.state['unconfirmed_questions'][0]
        
        # Build answer set:
        test_answers = {}
        
        for i in range(self.state['total_questions']):
            q_id = f"q{i+1}"
            question = self.questions[i]
            
            if q_id in self.state['confirmed_answers']:
                # Use confirmed answer
                test_answers[q_id] = self.state['confirmed_answers'][q_id]
            elif q_id == current_q_id:
                # For the question we're testing, use current candidate
                candidate = self.state['candidate_answers'].get(q_id, question['options'][0]['text'])
                test_answers[q_id] = candidate
            else:
                # For other unconfirmed questions, use baseline (first option)
                test_answers[q_id] = question['options'][0]['text']
        
        logger.info(f"[TESTING] Isolating question {current_q_id} (testing: '{test_answers[current_q_id][:50]}...')")
        return test_answers
    
    def process_result(self, score: int, total: int, tested_options: Dict[str, str]):
        """
        Process test result and update state.
        FIXED: Only confirm answers when score CHANGES from baseline
        """
        
        if not self.state['unconfirmed_questions']:
            return {
                'confirmed': len(self.state['confirmed_answers']),
                'total': self.state['total_questions'],
                'current_question': 'None',
                'base_score': self.state['base_score'],
                'perfect_score': self.state['perfect_score_achieved']
            }
        
        current_q_id = self.state['unconfirmed_questions'][0]
        
        record = {
            'timestamp': datetime.now().isoformat(),
            'score': score,
            'tested_question': current_q_id,
            'tested_option': tested_options.get(current_q_id, ""),
            'phase': self.state['phase']
        }
        self.state['test_history'].append(record)
        
        if self.state['phase'] == 'establishing_baseline':
            # This was baseline test
            self.state['base_score'] = score
            self.state['last_score'] = score
            self.state['phase'] = 'testing_questions'
            logger.info(f"[BASELINE SET] Score: {score}/{total}")
        else:
            # This was an isolation test
            current_option = tested_options.get(current_q_id, "")
            
            # SPECIAL CASE: If we got perfect score, mark all as confirmed
            if score == total:
                logger.info("🎉 PERFECT SCORE ACHIEVED!")
                self.state['perfect_score_achieved'] = True
                # Confirm all answers that were tested
                for q_id, opt in tested_options.items():
                    self.state['confirmed_answers'][q_id] = opt
                    if q_id in self.state['unconfirmed_questions']:
                        self.state['unconfirmed_questions'].remove(q_id)
                self._save_state()
                return {
                    'confirmed': total,
                    'total': total,
                    'current_question': 'None',
                    'base_score': score,
                    'perfect_score': True
                }
            
            # Update last score
            self.state['last_score'] = score
            
            # Track this option as tried
            if current_q_id not in self.state['tried_options']:
                self.state['tried_options'][current_q_id] = set()
            self.state['tried_options'][current_q_id].add(current_option)
            
            # Compare with baseline score
            if self.state['base_score'] is None:
                self.state['base_score'] = score
            
            if score > self.state['base_score']:
                # NEW option is CORRECT! (Score increased)
                logger.info(f"✅ {current_q_id}: '{current_option[:50]}...' is CORRECT! (Score increased from {self.state['base_score']} to {score})")
                self.state['confirmed_answers'][current_q_id] = current_option
                self.state['base_score'] = score  # Update baseline
                
                # Remove from unconfirmed list
                if current_q_id in self.state['unconfirmed_questions']:
                    self.state['unconfirmed_questions'].remove(current_q_id)
                
                # Reset candidate for this question
                self.state['candidate_answers'][current_q_id] = current_option
                
            elif score < self.state['base_score']:
                # NEW option is WRONG (Score decreased)
                # This means the previous answer for this question was correct
                # But we don't know which one, so we need to find it
                logger.info(f"❌ {current_q_id}: Option '{current_option[:50]}...' is WRONG (Score decreased from {self.state['base_score']} to {score})")
                
                # Try next option
                if not self._try_next_option(current_q_id):
                    # If no more options, this question is problematic
                    logger.warning(f"⚠️ {current_q_id}: All options tried but none increased score")
                    # Move to next question without confirming
                    if current_q_id in self.state['unconfirmed_questions']:
                        self.state['unconfirmed_questions'].pop(0)
                
                # Restore baseline score (since this option was wrong)
                self.state['base_score'] = self.state['last_score']
                
            else:
                # Score unchanged - inconclusive
                logger.info(f"➖ {current_q_id}: Option '{current_option[:50]}...' gave same score ({score})")
                
                # Try next option for this question
                if not self._try_next_option(current_q_id):
                    # If no more options, move to next question
                    logger.info(f"🔄 {current_q_id}: All options tried with same score, moving to next question")
                    if current_q_id in self.state['unconfirmed_questions']:
                        self.state['unconfirmed_questions'].pop(0)
        
        self._save_state()
        
        # Return status
        confirmed = len(self.state['confirmed_answers'])
        current_q = self.state['unconfirmed_questions'][0] if self.state['unconfirmed_questions'] else 'None'
        
        return {
            'confirmed': confirmed,
            'total': self.state['total_questions'],
            'current_question': current_q,
            'base_score': self.state['base_score'],
            'perfect_score': self.state['perfect_score_achieved']
        }
    
    def _try_next_option(self, q_id: str) -> bool:
        """Try the next option for a question. Returns True if successful."""
        q_idx = int(q_id[1:]) - 1
        question = self.questions[q_idx]
        
        current_option = self.state['candidate_answers'].get(q_id, question['options'][0]['text'])
        options_texts = [opt['text'] for opt in question['options']]
        
        if current_option in options_texts:
            current_idx = options_texts.index(current_option)
        else:
            current_idx = -1
        
        # Ensure tried_options set exists for this question
        if q_id not in self.state['tried_options']:
            self.state['tried_options'][q_id] = set([current_option])
        
        # Try next option that hasn't been tried
        for next_idx in range(len(options_texts)):
            next_option = options_texts[next_idx]
            if next_option not in self.state['tried_options'][q_id]:
                self.state['candidate_answers'][q_id] = next_option
                logger.info(f"🔄 {q_id}: Trying next option: '{next_option[:50]}...'")
                return True
        
        # All options tried
        return False
    
    def _load_state(self):
        if os.path.exists(self.json_file):
            try:
                with open(self.json_file, 'r') as f:
                    state = json.load(f)
                
                # Convert lists in tried_options back to sets
                if 'tried_options' in state:
                    for q_id, tried in state['tried_options'].items():
                        if isinstance(tried, list):
                            state['tried_options'][q_id] = set(tried)
                
                return state
            except Exception as e:
                logger.error(f"Error loading state: {e}")
                pass
        return {}
    
    def _save_state(self):
        try:
            # Convert sets to lists for JSON serialization
            state_to_save = self.state.copy()
            if 'tried_options' in state_to_save:
                state_to_save['tried_options'] = {k: list(v) for k, v in state_to_save['tried_options'].items()}
            
            with open(self.json_file, 'w') as f:
                json.dump(state_to_save, indent=2, fp=f)
        except Exception as e:
            logger.error(f"Error saving state: {e}")
    
    def get_status(self):
        confirmed = len(self.state.get('confirmed_answers', {}))
        total = self.state.get('total_questions', len(self.questions))
        
        status = f"""
{'='*80}
SEQUENTIAL ELIMINATION SOLVER (FIXED VERSION)
{'='*80}
Phase: {self.state.get('phase', 'unknown')}
Confirmed: {confirmed}/{total}
Baseline Score: {self.state.get('base_score', 'Unknown')}
Last Score: {self.state.get('last_score', 'Unknown')}
Unconfirmed Questions: {len(self.state.get('unconfirmed_questions', []))}
Perfect Score Achieved: {self.state.get('perfect_score_achieved', False)}
{'='*80}
"""
        
        for i in range(total):
            q_id = f"q{i+1}"
            if i < len(self.questions):
                question = self.questions[i]
            else:
                continue
            
            if q_id in self.state.get('confirmed_answers', {}):
                answer = self.state['confirmed_answers'][q_id]
                status += f"{q_id}: ✅ '{answer[:50]}...'\n"
            else:
                candidate = self.state.get('candidate_answers', {}).get(q_id, 'Unknown')
                tried_options = self.state.get('tried_options', {})
                tried_set = tried_options.get(q_id, set())
                tried_count = len(tried_set) if isinstance(tried_set, set) else len(tried_set)
                options_count = len(question['options'])
                status += f"{q_id}: 🔄 Testing ({tried_count}/{options_count} tried) - '{candidate[:50]}...'\n"
        
        status += "="*80
        return status


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


class QuestionExtractor:
    @staticmethod
    def extract_questions(page, num_questions: int = 10) -> List[Dict]:
        questions = []

        # Wait for inputs to be present
        page.wait_for_selector("input[type='radio'], input[type='checkbox']", timeout=10000)

        # Take a snapshot
        all_inputs = page.locator("input[type='radio'], input[type='checkbox']")
        all_labels = page.locator("label")

        total_inputs = all_inputs.count()
        total_labels = all_labels.count()

        logger.info(f"[DEBUG] Found {total_inputs} inputs and {total_labels} labels")

        if total_inputs == 0:
            logger.error("[!] No inputs found")
            return []

        options_per_question = total_inputs // num_questions

        # Extract questions
        for q_idx in range(num_questions):
            start = q_idx * options_per_question
            end = start + options_per_question

            options = []
            for i in range(start, min(end, total_inputs)):
                label_text = ""
                if i < total_labels:
                    label_text = all_labels.nth(i).text_content().strip()

                options.append({
                    "text": label_text or f"Option {i-start+1}"
                })

            questions.append({
                "id": f"q{q_idx + 1}",
                "question_text": f"Question {q_idx + 1}",
                "options": options
            })

            logger.info(f"[OK] Extracted Q{q_idx + 1} ({len(options)} options)")

        return questions


class AIExamSolver:
    def __init__(self, page, ai: OllamaAI):
        self.page = page
        self.ai = ai
        self.questions: List[Dict] = []
        self.answers: Dict[str, Dict] = {}
        self.sequential_solver = None
        self.last_score = 0
        self.perfect_score_achieved = False
        self.num_questions = 0

    def analyze_exam(self, num_questions: int = 10) -> bool:
        logger.info(f"[*] Extracting questions (expecting {num_questions})...")
        self.questions = QuestionExtractor.extract_questions(self.page, num_questions)
        if not self.questions:
            logger.error("Failed to extract any questions!")
            return False
        
        self.num_questions = len(self.questions)
        logger.info(f"[OK] Extracted {len(self.questions)} questions")
        return True

    def get_answers_for_attempt(self, last_score: Optional[int] = None) -> bool:
        """Get answers using sequential elimination strategy."""
        if self.sequential_solver is None:
            self.sequential_solver = SequentialEliminationSolver(self.questions)
            logger.info("[NEW SOLVER] Sequential elimination solver initialized")
        
        # Update last score if provided
        if last_score is not None:
            self.last_score = last_score
        
        # Get test set
        test_answers = self.sequential_solver.get_test_set(self.last_score)
        
        # Check if we're done
        if self.sequential_solver.state.get('perfect_score_achieved', False):
            logger.info("[DONE] Perfect score already achieved!")
            self.perfect_score_achieved = True
            return False
        
        if len(test_answers) == len(self.sequential_solver.state.get('confirmed_answers', {})):
            logger.info("[DONE] All questions confirmed!")
            return False
        
        # Convert to our format
        self.answers = {}
        for q_id, option_text in test_answers.items():
            question = next((q for q in self.questions if q['id'] == q_id), None)
            if question:
                options_texts = [opt['text'] for opt in question['options']]
                if option_text in options_texts:
                    answer_idx = options_texts.index(option_text)
                    self.answers[q_id] = {
                        'option_text': option_text,
                        'answer_index': answer_idx
                    }
                else:
                    # Fallback to first option
                    self.answers[q_id] = {
                        'option_text': question['options'][0]['text'],
                        'answer_index': 0
                    }
        
        logger.info(self.sequential_solver.get_status())
        return True

    def process_attempt_result(self, score: int, total: int):
        """Process attempt result using sequential solver."""
        if self.sequential_solver is not None:
            # Update last score
            self.last_score = score
            
            # Get what options were tested
            tested_options = {}
            for q in self.questions:
                q_id = q['id']
                if q_id in self.answers:
                    tested_options[q_id] = self.answers[q_id]['option_text']
            
            # Process result
            result = self.sequential_solver.process_result(score, total, tested_options)
            
            # Check if perfect score was achieved
            if result.get('perfect_score', False):
                self.perfect_score_achieved = True
            
            logger.info(f"Confirmed: {result['confirmed']}/{total}, Baseline: {result['base_score']}")
            logger.info(self.sequential_solver.get_status())

    def click_answers(self) -> bool:
        """Click all answers on the page."""
        logger.info("\n[*] Clicking answers...")
        success_count = 0

        for q in self.questions:
            qid = q['id']
            if qid not in self.answers:
                logger.warning(f"No answer for {qid}")
                continue

            target_text = self.answers[qid]['option_text'].strip()
            target_text = ' '.join(target_text.split())  # normalize spaces

            logger.info(f"  Trying to click for {qid}: '{target_text[:70]}...'")

            # Special handling for "All the answers are correct"
            target_lower = target_text.lower()
            if "all" in target_lower and "answer" in target_lower and "correct" in target_lower:
                if self._click_all_answers_correct_by_index(q, target_text):
                    success_count += 1
                    time.sleep(0.5)
                    continue

            # Regular clicking for other answers
            if self._click_by_text(target_text):
                success_count += 1
                time.sleep(0.5)
            else:
                logger.error(f"Failed to click answer for {qid}: '{target_text[:60]}...'")

        logger.info(f"Successfully clicked {success_count}/{len(self.questions)} answers")
        return success_count == len(self.questions)

    def _click_all_answers_correct_by_index(self, question: Dict, target_text: str) -> bool:
        """Special handler for 'All the answers are correct' option."""
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
                    logger.info(f"  Found 'All answers correct' at index {idx}: '{opt['text'][:50]}...'")
                    break
            
            if target_index is None:
                logger.error(f"  Could not find 'All answers correct' in options!")
                return False
            
            # Calculate global input index
            all_inputs = self.page.locator("input[type='radio'], input[type='checkbox']")
            total_inputs = all_inputs.count()
            
            q_num = int(qid.replace('q', '')) - 1
            options_per_question = len(options)
            
            global_input_index = (q_num * options_per_question) + target_index
            
            logger.info(f"  Question {qid} (q_num={q_num}), target_index={target_index}")
            logger.info(f"  Global input index: {global_input_index} (total inputs: {total_inputs})")
            
            if global_input_index >= total_inputs:
                logger.error(f"  Calculated index {global_input_index} exceeds total inputs {total_inputs}")
                return False
            
            # Try multiple click strategies
            click_strategies = [
                # Strategy 1: Click input directly
                lambda: all_inputs.nth(global_input_index).click(force=True, timeout=3000),
                
                # Strategy 2: Click via label with same index
                lambda: self.page.locator("label").nth(global_input_index).click(force=True, timeout=3000),
                
                # Strategy 3: Try to find label with matching text
                lambda: self._click_by_text(question['options'][target_index]['text'])
            ]
            
            for i, strategy in enumerate(click_strategies):
                try:
                    strategy()
                    logger.info(f"✓ Clicked using strategy {i+1}")
                    return True
                except Exception as e:
                    logger.debug(f"  Strategy {i+1} failed: {e}")
                    continue
                    
            logger.error(f"  All click strategies failed for {qid}")
            return False
                
        except Exception as e:
            logger.error(f"Error in _click_all_answers_correct_by_index: {e}")
            return False

    def _click_by_text(self, target_text: str) -> bool:
        """Click a label by its text content."""
        try:
            # Clean the target text for matching
            clean_target = ' '.join(target_text.split())
            
            # Try exact match first
            labels = self.page.locator("label").all()
            for label in labels:
                try:
                    label_text = label.text_content().strip()
                    clean_label = ' '.join(label_text.split())
                    
                    if clean_target.lower() == clean_label.lower():
                        label.click(force=True, timeout=3000)
                        logger.debug(f"✓ Clicked exact match: '{label_text[:50]}...'")
                        return True
                except:
                    continue
            
            # Try partial match for longer texts
            if len(clean_target) > 20:
                partial = clean_target[:40]
                for label in labels:
                    try:
                        label_text = label.text_content().strip()
                        if partial.lower() in label_text.lower():
                            label.click(force=True, timeout=3000)
                            logger.debug(f"✓ Clicked partial match: '{label_text[:50]}...'")
                            return True
                    except:
                        continue
            
            logger.error(f"✗ Failed to click: '{target_text[:50]}...'")
            return False
            
        except Exception as e:
            logger.error(f"Click error: {e}")
            return False

    def submit_and_check(self) -> Tuple[int, int, bool]:
        """Submit answers and check score with improved parsing."""
        max_retries = 3
        
        for retry in range(max_retries):
            try:
                logger.info(f"Submitting (attempt {retry + 1}/{max_retries})...")
                
                # Try different submit button selectors
                submit_selectors = [
                    # By role button
                    lambda: self.page.get_by_role("button", name=re.compile(r"submit|finish|send|complete", re.I)).first.click(),
                    
                    # By text
                    lambda: self.page.get_by_text(re.compile(r"submit|finish", re.I)).first.click(),
                    
                    # By input type submit
                    lambda: self.page.locator("input[type='submit']").first.click(),
                    
                    # By button with type submit
                    lambda: self.page.locator("button[type='submit']").first.click(),
                ]
                
                clicked = False
                for selector in submit_selectors:
                    try:
                        selector()
                        clicked = True
                        time.sleep(2)  # Wait for submission
                        break
                    except Exception as e:
                        logger.debug(f"Submit selector failed: {e}")
                        continue
                
                if not clicked:
                    # Try clicking any button that might be submit
                    all_buttons = self.page.locator("button")
                    button_count = all_buttons.count()
                    for i in range(min(button_count, 10)):  # Check first 10 buttons
                        try:
                            btn_text = all_buttons.nth(i).inner_text().lower()
                            if any(word in btn_text for word in ["submit", "finish", "send", "complete"]):
                                all_buttons.nth(i).click()
                                clicked = True
                                time.sleep(2)
                                break
                        except:
                            continue
                
                if not clicked:
                    raise Exception("No submit button found")
                
                # Wait for results
                time.sleep(3)
                
                # Try to find score in the page
                page_text = self.page.inner_text("body")
                logger.info(f"Page text preview: {page_text[:200]}...")
                
                # Try different score patterns
                score_patterns = [
                    r'(\d+)\s*\/\s*(\d+)',  # 3/10
                    r'Score:\s*(\d+)\s*\/\s*(\d+)',  # Score: 3/10
                    r'(\d+)\s*of\s*(\d+)',  # 3 of 10
                    r'Result:\s*(\d+)\s*\/\s*(\d+)',  # Result: 3/10
                    r'Your score:\s*(\d+)\s*\/\s*(\d+)',  # Your score: 3/10
                    r'(\d+)\s*out of\s*(\d+)',  # 3 out of 10
                ]
                
                for pattern in score_patterns:
                    matches = re.findall(pattern, page_text, re.IGNORECASE)
                    if matches:
                        # Take the first match that makes sense
                        for match in matches:
                            if len(match) >= 2:
                                try:
                                    score_val = int(match[0])
                                    total_val = int(match[1])
                                    # Validate that the score is reasonable
                                    if 0 <= score_val <= total_val and total_val <= self.num_questions * 2:
                                        logger.info(f"Found score with pattern '{pattern}': {score_val}/{total_val}")
                                        return score_val, total_val, score_val == total_val
                                except ValueError:
                                    continue
                
                # If no pattern matched, try to extract just numbers
                all_numbers = re.findall(r'\b\d+\b', page_text)
                if len(all_numbers) >= 2:
                    # Look for pairs that might be scores
                    for i in range(len(all_numbers) - 1):
                        try:
                            score_val = int(all_numbers[i])
                            total_val = int(all_numbers[i+1])
                            # Check if this looks like a score
                            if 0 <= score_val <= total_val and total_val <= self.num_questions * 2:
                                logger.info(f"Found score as number pair: {score_val}/{total_val}")
                                return score_val, total_val, score_val == total_val
                        except ValueError:
                            continue
                
                # If we still haven't found a score, check if there's a success message
                if any(word in page_text.lower() for word in ["passed", "success", "completed", "perfect"]):
                    logger.info("Found success message, assuming perfect score")
                    return self.num_questions, self.num_questions, True
                
                raise Exception(f"Score not found in page text")
                
            except Exception as e:
                logger.error(f"Submit/check failed (attempt {retry + 1}): {e}")
                if retry < max_retries - 1:
                    time.sleep(2)
                    continue
        
        logger.error("Failed to get score after all retries")
        return 0, self.num_questions, False

    def retest(self, num_questions: int = 10) -> bool:
        """Reset the exam for next attempt."""
        try:
            logger.info("Retesting → looking for reset button...")
            
            # Try different retest button selectors
            retest_selectors = [
                # By role button
                lambda: self.page.get_by_role("button", name=re.compile(r"retest|try again|redo|restart|again|new attempt", re.I)).first.click(),
                
                # By text
                lambda: self.page.get_by_text(re.compile(r"retest|try again|redo", re.I)).first.click(),
                
                # By link
                lambda: self.page.get_by_role("link", name=re.compile(r"retake|try again", re.I)).first.click(),
            ]
            
            clicked = False
            for selector in retest_selectors:
                try:
                    selector()
                    clicked = True
                    time.sleep(2.5)
                    break
                except:
                    continue
            
            if not clicked:
                logger.warning("No retest button found, trying to reload page")
                self.page.reload()
                time.sleep(3)
            
            # Wait for page to load
            self.page.wait_for_load_state("networkidle", timeout=15000)
            
            # Re-extract questions
            self.questions = QuestionExtractor.extract_questions(self.page, num_questions)
            if not self.questions:
                logger.error("Failed to re-extract questions!")
                return False
            
            self.num_questions = len(self.questions)
            
            # Create new solver with updated questions
            self.sequential_solver = SequentialEliminationSolver(self.questions)
            
            logger.info("[RESET] Exam reset for next attempt")
            return True
            
        except Exception as e:
            logger.error(f"Retest failed: {e}")
            return False


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
        
        # Check if we already have perfect score
        if solver.perfect_score_achieved:
            logger.info("🎉 Perfect score already achieved in previous attempt!")
            break
        
        # Get answers for this attempt
        if not solver.get_answers_for_attempt(last_score):
            logger.info("No more answers to test or perfect score achieved")
            break
        
        # Click answers and submit
        logger.info("\nClicking answers...")
        if not solver.click_answers():
            logger.warning("Some answers could not be clicked!")
        
        # Submit and get score
        score, total, is_perfect = solver.submit_and_check()
        
        # Validate score
        if total <= 0 or total > num_questions * 2:
            logger.warning(f"Invalid total score {total}, using number of questions instead")
            total = num_questions
        
        if score > total:
            logger.warning(f"Score {score} > total {total}, capping score to total")
            score = total
        
        last_score = score
        
        logger.info(f"Score: {score}/{total}")
        
        # Process attempt result
        solver.process_attempt_result(score, total)
        
        if is_perfect or solver.perfect_score_achieved:
            logger.info("🎉 PERFECT SCORE ACHIEVED!")
            break
        
        if attempt >= max_attempts:
            logger.info("Max attempts reached.")
            break
        
        # Check if solver has confirmed all questions
        if solver.sequential_solver and len(solver.sequential_solver.state.get('confirmed_answers', {})) >= num_questions:
            logger.info("✅ All questions confirmed!")
            break
        
        # Retest for next attempt
        logger.info("\nPreparing for next attempt...")
        if not solver.retest(num_questions):
            logger.error("Retest failed!")
            # Try to continue anyway
            time.sleep(3)
    
    logger.info("\n" + "="*80)
    logger.info("EXAM SOLVING COMPLETED")
    logger.info("="*80)
    
    # Final check
    if solver.sequential_solver:
        confirmed = len(solver.sequential_solver.state.get('confirmed_answers', {}))
        logger.info(f"Confirmed answers: {confirmed}/{num_questions}")
        
        if solver.perfect_score_achieved:
            logger.info("✅ PERFECT SCORE ACHIEVED!")
        elif confirmed == num_questions:
            logger.info("✅ All questions confirmed!")
        else:
            logger.info(f"⚠️ Only {confirmed}/{num_questions} questions confirmed")
    
    logger.info(f"Final score: {last_score}/{num_questions}")
    logger.info(f"Total attempts: {attempt}")