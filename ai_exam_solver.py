"""AI-Powered Exam Solver using Ollama llama3.1:8b - IMPROVED VERSION"""

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
    IMPROVED SOLVER - Only confirms answers when perfect score is achieved.
    Uses systematic testing with best-answer tracking.
    """
    
    def __init__(self, questions: List[Dict], json_file: str = "improved_progress.json"):
        self.questions = questions
        self.json_file = json_file
        self.state = self._load_state()
        
        if not self.state.get('initialized'):
            self._initialize_state()
        else:
            self._migrate_state()
    
    def _initialize_state(self):
        """Initialize fresh state."""
        self.state = {
            'initialized': True,
            'start_time': datetime.now().isoformat(),
            'total_questions': len(self.questions),
            'confirmed_answers': {},      # Only when perfect score achieved
            'best_answers': {},           # Current best known combination
            'best_score': 0,              # Best score achieved so far
            'question_scores': {},        # {q_id: {option: best_score_with_this_option}}
            'test_history': [],
            'perfect_score_achieved': False,
            'last_score': None,
            'unconfirmed_questions': [f"q{i+1}" for i in range(len(self.questions))]
        }
        
        # Start with all first options
        for question in self.questions:
            q_id = question['id']
            self.state['best_answers'][q_id] = question['options'][0]['text']
            self.state['question_scores'][q_id] = {}
        
        self._save_state()
    
    def _migrate_state(self):
        """Ensure state has all required fields."""
        defaults = {
            'confirmed_answers': {},
            'best_answers': {},
            'best_score': 0,
            'question_scores': {},
            'test_history': [],
            'perfect_score_achieved': False,
            'last_score': None,
            'unconfirmed_questions': [f"q{i+1}" for i in range(len(self.questions))]
        }
        
        for key, default_value in defaults.items():
            if key not in self.state:
                self.state[key] = default_value
        
        # Ensure question_scores has entries for all questions
        for question in self.questions:
            q_id = question['id']
            if q_id not in self.state.get('question_scores', {}):
                self.state['question_scores'][q_id] = {}
            if q_id not in self.state.get('best_answers', {}):
                self.state['best_answers'][q_id] = question['options'][0]['text']
    
    def get_test_set(self, current_score: Optional[int] = None) -> Dict[str, str]:
        """Get next answer combination to test using exhaustive search."""
        
        # Update last score
        if current_score is not None:
            self.state['last_score'] = current_score
            
            # Update best score if this is better
            if current_score > self.state['best_score']:
                self.state['best_score'] = current_score
                logger.info(f"🎯 NEW BEST SCORE: {current_score}/{self.state['total_questions']}")
        
        # Check if we already have perfect score
        if self.state['perfect_score_achieved']:
            logger.info("✅ Perfect score already achieved!")
            return self.state['confirmed_answers']
        
        # If all confirmed, return them
        if len(self.state['confirmed_answers']) == self.state['total_questions']:
            return self.state['confirmed_answers']
        
        # Start with current best answers
        test_answers = self.state['best_answers'].copy()
        
        # Find which question to vary next
        # Strategy: For each question, try all options and pick the one that gives best score
        focus_q_id = None
        next_option = None
        
        # Try to find a question with untried options
        for i in range(self.state['total_questions']):
            q_id = f"q{i+1}"
            question = next((q for q in self.questions if q['id'] == q_id), None)
            if not question:
                continue
            
            # Get all options for this question
            all_options = [opt['text'] for opt in question['options']]
            tried_options = set(self.state['question_scores'].get(q_id, {}).keys())
            
            # Find untried options
            untried = [opt for opt in all_options if opt not in tried_options]
            
            if untried:
                focus_q_id = q_id
                next_option = untried[0]
                test_answers[q_id] = next_option
                logger.info(f"🔄 Testing {q_id} with option {len(tried_options)+1}/{len(all_options)}: '{next_option[:50]}...'")
                break
        
        # If all questions have been tried with all options, pick best combination
        if focus_q_id is None:
            logger.info("⚠️ All options tried individually. Starting combination testing...")
            
            # Generate a new combination to test
            # Use a counter to systematically try different combinations
            if 'combination_counter' not in self.state:
                self.state['combination_counter'] = 0
            
            self.state['combination_counter'] += 1
            counter = self.state['combination_counter']
            
            # Convert counter to base-4 to generate different combinations
            for i in range(self.state['total_questions']):
                q_id = f"q{i+1}"
                question = next((q for q in self.questions if q['id'] == q_id), None)
                if question:
                    num_options = len(question['options'])
                    option_index = (counter // (num_options ** i)) % num_options
                    test_answers[q_id] = question['options'][option_index]['text']
            
            logger.info(f"🔄 Testing combination #{counter}")
        
        return test_answers
    
    def _get_next_question_to_test(self) -> Optional[str]:
        """Find which question to test next - prioritize least-tested questions."""
        # Find questions that haven't been fully explored
        min_tested = float('inf')
        next_q = None
        
        for i in range(self.state['total_questions']):
            q_id = f"q{i+1}"
            if q_id not in self.state['confirmed_answers']:
                question = next((q for q in self.questions if q['id'] == q_id), None)
                if question:
                    tried_count = len(self.state['question_scores'].get(q_id, {}))
                    total_options = len(question['options'])
                    
                    # Prioritize questions with fewer tested options
                    if tried_count < total_options and tried_count < min_tested:
                        min_tested = tried_count
                        next_q = q_id
        
        # If all questions fully tested, return first unconfirmed
        if next_q is None:
            for i in range(self.state['total_questions']):
                q_id = f"q{i+1}"
                if q_id not in self.state['confirmed_answers']:
                    return q_id
        
        return next_q
    
    def _get_next_option_to_try(self, q_id: str) -> Optional[str]:
        """Get next option to try for a question."""
        question = next((q for q in self.questions if q['id'] == q_id), None)
        if not question:
            return None
        
        # Find options we haven't tried yet
        tried_options = set(self.state['question_scores'].get(q_id, {}).keys())
        all_options = [opt['text'] for opt in question['options']]
        
        # Return first untried option
        for option in all_options:
            if option not in tried_options:
                return option
        
        # All options tried - return the one with best score
        if self.state['question_scores'].get(q_id):
            best_option = max(
                self.state['question_scores'][q_id].items(),
                key=lambda x: x[1]
            )[0]
            return best_option
        
        return all_options[0]
    
    def process_result(self, score: int, total: int, tested_options: Dict[str, str]):
        """Process test result - ONLY confirm when perfect score achieved."""
        
        # Record test
        record = {
            'timestamp': datetime.now().isoformat(),
            'score': score,
            'total': total,
            'tested_options': {k: v[:50] + '...' if len(v) > 50 else v 
                             for k, v in tested_options.items()}
        }
        self.state['test_history'].append(record)
        
        # Update last score
        self.state['last_score'] = score
        
        # CHECK FOR PERFECT SCORE - This is the ONLY way to confirm answers
        if score == total:
            logger.info("🎉 PERFECT SCORE ACHIEVED! All answers are now CONFIRMED!")
            self.state['perfect_score_achieved'] = True
            self.state['confirmed_answers'] = tested_options.copy()
            self.state['best_answers'] = tested_options.copy()
            self.state['best_score'] = score
            self.state['unconfirmed_questions'] = []
            self._save_state()
            return {
                'confirmed': total,
                'total': total,
                'best_score': score,
                'perfect_score': True
            }
        
        # CRITICAL FIX: Track EVERY option that was tested, not just changed ones
        # Record score for ALL tested options
        for q_id, option in tested_options.items():
            if q_id not in self.state['question_scores']:
                self.state['question_scores'][q_id] = {}
            
            # Only record if this is a new option we haven't tried
            if option not in self.state['question_scores'][q_id]:
                self.state['question_scores'][q_id][option] = score
                
                question = next((q for q in self.questions if q['id'] == q_id), None)
                if question:
                    tried = len(self.state['question_scores'][q_id])
                    total_opts = len(question['options'])
                    logger.info(f"📊 {q_id}: '{option[:30]}...' → score {score} ({tried}/{total_opts} options tried)")
        
        # Update best score and answers if this is an improvement
        if score > self.state['best_score']:
            self.state['best_score'] = score
            self.state['best_answers'] = tested_options.copy()
            logger.info(f"📈 New best score: {score}/{total}!")
        
        self._save_state()
        
        return {
            'confirmed': len(self.state['confirmed_answers']),
            'total': self.state['total_questions'],
            'best_score': self.state['best_score'],
            'perfect_score': self.state['perfect_score_achieved']
        }
    
    def _load_state(self):
        if os.path.exists(self.json_file):
            try:
                with open(self.json_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading state: {e}")
        return {}
    
    def _save_state(self):
        try:
            with open(self.json_file, 'w') as f:
                json.dump(self.state, indent=2, fp=f)
        except Exception as e:
            logger.error(f"Error saving state: {e}")
    
    def get_status(self):
        confirmed = len(self.state.get('confirmed_answers', {}))
        total = self.state.get('total_questions', len(self.questions))
        best_score = self.state.get('best_score', 0)
        
        status = f"""
{'='*80}
IMPROVED SEQUENTIAL SOLVER
{'='*80}
Perfect Score: {'✅ YES' if self.state.get('perfect_score_achieved', False) else '❌ NO'}
Best Score: {best_score}/{total}
Last Score: {self.state.get('last_score', 'N/A')}
Confirmed: {confirmed}/{total}
{'='*80}
"""
        
        # Show current best answers
        for i in range(total):
            q_id = f"q{i+1}"
            if q_id in self.state.get('confirmed_answers', {}):
                answer = self.state['confirmed_answers'][q_id]
                status += f"{q_id}: ✅ CONFIRMED '{answer[:50]}...'\n"
            elif q_id in self.state.get('best_answers', {}):
                answer = self.state['best_answers'][q_id]
                tried = len(self.state.get('question_scores', {}).get(q_id, {}))
                question = next((q for q in self.questions if q['id'] == q_id), None)
                total_opts = len(question['options']) if question else 0
                
                # Show score info for this question's options
                q_scores = self.state.get('question_scores', {}).get(q_id, {})
                if q_scores:
                    best_opt_score = max(q_scores.values()) if q_scores else 0
                    status += f"{q_id}: 🔄 TESTING ({tried}/{total_opts} tried, best: {best_opt_score}) '{answer[:40]}...'\n"
                else:
                    status += f"{q_id}: 🔄 TESTING ({tried}/{total_opts} tried) '{answer[:50]}...'\n"
        
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
            logger.info("[NEW SOLVER] Improved sequential solver initialized")
        
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
            
            logger.info(f"Confirmed: {result['confirmed']}/{total}, Best Score: {result['best_score']}")
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
                    time.sleep(5)
                    continue

            # Regular clicking for other answers
            if self._click_by_text(target_text):
                success_count += 1
                time.sleep(5)
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
                    lambda: self.page.get_by_role("button", name=re.compile(r"submit|finish|send|complete", re.I)).first.click(),
                    lambda: self.page.get_by_text(re.compile(r"submit|finish", re.I)).first.click(),
                    lambda: self.page.locator("input[type='submit']").first.click(),
                    lambda: self.page.locator("button[type='submit']").first.click(),
                ]
                
                clicked = False
                for selector in submit_selectors:
                    try:
                        selector()
                        clicked = True
                        time.sleep(5)
                        break
                    except Exception as e:
                        logger.debug(f"Submit selector failed: {e}")
                        continue
                
                if not clicked:
                    all_buttons = self.page.locator("button")
                    button_count = all_buttons.count()
                    for i in range(min(button_count, 10)):
                        try:
                            btn_text = all_buttons.nth(i).inner_text().lower()
                            if any(word in btn_text for word in ["submit", "finish", "send", "complete"]):
                                all_buttons.nth(i).click()
                                clicked = True
                                time.sleep(5)
                                break
                        except:
                            continue
                
                if not clicked:
                    raise Exception("No submit button found")
                
                # Wait for results
                time.sleep(5)
                
                # Try to find score in the page
                page_text = self.page.inner_text("body")
                logger.info(f"Page text preview: {page_text[:200]}...")
                
                # Try different score patterns
                score_patterns = [
                    r'(\d+)\s*\/\s*(\d+)',
                    r'Score:\s*(\d+)\s*\/\s*(\d+)',
                    r'(\d+)\s*of\s*(\d+)',
                    r'Result:\s*(\d+)\s*\/\s*(\d+)',
                    r'Your score:\s*(\d+)\s*\/\s*(\d+)',
                    r'(\d+)\s*out of\s*(\d+)',
                ]
                
                for pattern in score_patterns:
                    matches = re.findall(pattern, page_text, re.IGNORECASE)
                    if matches:
                        for match in matches:
                            if len(match) >= 2:
                                try:
                                    score_val = int(match[0])
                                    total_val = int(match[1])
                                    if 0 <= score_val <= total_val and total_val <= self.num_questions * 2:
                                        logger.info(f"Found score with pattern '{pattern}': {score_val}/{total_val}")
                                        return score_val, total_val, score_val == total_val
                                except ValueError:
                                    continue
                
                # Extract just numbers
                all_numbers = re.findall(r'\b\d+\b', page_text)
                if len(all_numbers) >= 2:
                    for i in range(len(all_numbers) - 1):
                        try:
                            score_val = int(all_numbers[i])
                            total_val = int(all_numbers[i+1])
                            if 0 <= score_val <= total_val and total_val <= self.num_questions * 2:
                                logger.info(f"Found score as number pair: {score_val}/{total_val}")
                                return score_val, total_val, score_val == total_val
                        except ValueError:
                            continue
                
                # Check for success message
                if any(word in page_text.lower() for word in ["passed", "success", "completed", "perfect"]):
                    logger.info("Found success message, assuming perfect score")
                    return self.num_questions, self.num_questions, True
                
                raise Exception(f"Score not found in page text")
                
            except Exception as e:
                logger.error(f"Submit/check failed (attempt {retry + 1}): {e}")
                if retry < max_retries - 1:
                    time.sleep(5)
                    continue
        
        logger.error("Failed to get score after all retries")
        return 0, self.num_questions, False

    def retest(self, num_questions: int = 10) -> bool:
        """Reset the exam for next attempt."""
        try:
            logger.info("Retesting → looking for reset button...")
            
            retest_selectors = [
                lambda: self.page.get_by_role("button", name=re.compile(r"retest|try again|redo|restart|again|new attempt", re.I)).first.click(),
                lambda: self.page.get_by_text(re.compile(r"retest|try again|redo", re.I)).first.click(),
                lambda: self.page.get_by_role("link", name=re.compile(r"retake|try again", re.I)).first.click(),
            ]
            
            time.sleep(5)


            clicked = False
            for selector in retest_selectors:
                try:
                    selector()
                    clicked = True
                    time.sleep(5)
                    break
                except:
                    continue
            
            if not clicked:
                logger.warning("No retest button found, trying to reload page")
                self.page.reload()
                time.sleep(5)
            
            # Wait for page to load
            self.page.wait_for_load_state("networkidle", timeout=30000)

            # CRITICAL: Wait for quiz elements to actually appear
            logger.info("Waiting for quiz elements to appear...")
            try:
                self.page.wait_for_selector(
                    "input[type='radio'], input[type='checkbox']", 
                    state="visible",
                    timeout=30000
                )
                logger.info("✓ Quiz elements detected")
                
                # Extra wait to ensure all elements are rendered
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"Quiz elements did not appear: {e}")
                return False
            
            # Re-extract questions
            self.questions = QuestionExtractor.extract_questions(self.page, num_questions)
            if not self.questions:
                logger.error("Failed to re-extract questions!")
                return False
            
            self.num_questions = len(self.questions)
            
            # Keep existing solver (don't create new one)
            logger.info("[RESET] Exam reset for next attempt")
            return True
            
        except Exception as e:
            logger.error(f"Retest failed: {e}")
            return False


def run_ai_solver(page, num_questions: int = 10, max_attempts: int = 100):
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
        
        # Retest for next attempt
        logger.info("\nPreparing for next attempt...")
        if not solver.retest(num_questions):
            logger.error("Retest failed!")
            time.sleep(5)
    
    logger.info("\n" + "="*80)
    logger.info("EXAM SOLVING COMPLETED")
    logger.info("="*80)
    
    # Final check
    if solver.sequential_solver:
        confirmed = len(solver.sequential_solver.state.get('confirmed_answers', {}))
        logger.info(f"Confirmed answers: {confirmed}/{num_questions}")
        
        if solver.perfect_score_achieved:
            logger.info("✅ PERFECT SCORE ACHIEVED!")
        else:
            logger.info(f"⚠️ Best score achieved: {solver.sequential_solver.state.get('best_score', 0)}/{num_questions}")
    
    logger.info(f"Final score: {last_score}/{num_questions}")
    logger.info(f"Total attempts: {attempt}")