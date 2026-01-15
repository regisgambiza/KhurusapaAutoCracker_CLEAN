"""AI-Powered Exam Solver using Ollama llama3.1:8b"""

import logging
import json
import re
import time
from typing import Dict, List, Tuple, Optional
import requests

logger = logging.getLogger(__name__)


class OllamaAI:
    """Interface to Ollama AI for question answering."""
    
    def __init__(self, model: str = "llama3.1:8b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
    
    def ask(self, prompt: str, temperature: float = 0.3) -> str:
        """
        Send a prompt to Ollama and get response.
        
        Args:
            prompt: The prompt to send
            temperature: Sampling temperature (lower = more deterministic)
            
        Returns:
            AI response text
        """
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "temperature": temperature
            }
            
            response = requests.post(self.api_url, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "").strip()
            
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            return ""
    
    def answer_question(self, question: str, options: List[str]) -> Tuple[int, str]:
        """
        Use AI to answer a multiple choice question.
        
        Args:
            question: The question text
            options: List of answer options
            
        Returns:
            Tuple of (selected_index, reasoning)
        """
        # Format options with letters
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
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                answer_letter = result.get("answer", "A").upper().strip()
                reasoning = result.get("reasoning", "No reasoning provided")
                
                # Convert letter to index
                answer_index = ord(answer_letter) - 65
                if 0 <= answer_index < len(options):
                    return answer_index, reasoning
            
            # Fallback: look for letter in response
            for i, letter in enumerate([chr(65+j) for j in range(len(options))]):
                if letter in response[:10]:  # Check first part of response
                    return i, "Extracted from response"
            
        except Exception as e:
            logger.error(f"Failed to parse AI response: {e}")
        
        # Ultimate fallback
        return 0, "Default to first option"


class QuestionExtractor:
    """Extract questions and answers from exam page."""
    
    @staticmethod
    def extract_questions(page, num_questions: int = 10) -> List[Dict]:
        """
        Extract all questions and their options from the page.
        
        Args:
            page: Playwright page object
            num_questions: Expected number of questions (default 10)
        
        Returns:
            List of dicts with question data
        """
        questions = []
        
        try:
            # Find ALL radio/checkbox inputs on the page
            all_inputs = page.locator("input[type='radio'], input[type='checkbox']")
            all_labels = page.locator("label")
            total_inputs = all_inputs.count()
            total_labels = all_labels.count()
            
            logger.info(f"[DEBUG] Found {total_inputs} input elements and {total_labels} labels")
            
            if total_inputs == 0:
                logger.error("[!] No radio/checkbox inputs found on page!")
                with open("exam_page.html", "w", encoding="utf-8") as f:
                    f.write(page.content())
                return []
            
            # Calculate options per question
            options_per_question = total_inputs // num_questions
            logger.info(f"[DEBUG] Assuming {options_per_question} options per question ({total_inputs} / {num_questions})")
            
            # Extract questions by dividing inputs into groups
            for q_idx in range(num_questions):
                start_idx = q_idx * options_per_question
                end_idx = start_idx + options_per_question
                
                # Handle last question taking remaining inputs
                if q_idx == num_questions - 1:
                    end_idx = total_inputs
                
                # Get indices for this question
                indices = list(range(start_idx, end_idx))
                
                # Extract question text from first option's context
                first_input = all_inputs.nth(start_idx)
                question_text = QuestionExtractor._extract_question_near_input(first_input, page)
                
                if not question_text:
                    question_text = f"Question {q_idx + 1}"
                
                # Extract options for this question
                options = []
                for opt_idx, global_idx in enumerate(indices):
                    input_elem = all_inputs.nth(global_idx)
                    
                    # Get the label text (answer option text)
                    label_elem = all_labels.nth(global_idx) if global_idx < total_labels else None
                    option_text = label_elem.text_content().strip() if label_elem else f"Option {opt_idx + 1}"
                    
                    option_id = f"q{q_idx + 1}_opt{opt_idx + 1}"
                    
                    # Get input attributes
                    try:
                        input_id = input_elem.get_attribute('id')
                        input_name = input_elem.get_attribute('name')
                        input_value = input_elem.get_attribute('value')
                    except:
                        input_id = None
                        input_name = None
                        input_value = None
                    
                    option_data = {
                        'id': option_id,
                        'text': option_text,
                        'input_id': input_id,
                        'input_name': input_name,
                        'input_value': input_value,
                        'index': opt_idx,
                        'global_index': global_idx
                    }
                    
                    options.append(option_data)
                
                question_data = {
                    'id': f'q{q_idx + 1}',
                    'question_text': question_text,
                    'options': options
                }
                
                questions.append(question_data)
                logger.info(f"[OK] Q{q_idx + 1}: '{question_text[:50]}...' with {len(options)} options")
            
            return questions
            
        except Exception as e:
            logger.error(f"Failed to extract questions: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    @staticmethod
    def _extract_question_near_input(input_elem, page) -> str:
        """Extract question text by looking near an input element."""
        try:
            # Try to find parent container with question text
            parent_selectors = [
                'xpath=ancestor::div[contains(@class, "question")]',
                'xpath=ancestor::fieldset',
                'xpath=ancestor::div[@class]',
                'xpath=ancestor::*[contains(@class, "form")]'
            ]
            
            for selector in parent_selectors:
                try:
                    parent = input_elem.locator(selector).first
                    # Get text but try to filter out option texts
                    text = parent.text_content().strip()
                    lines = [l.strip() for l in text.split('\n') if len(l.strip()) > 10]
                    if lines:
                        return lines[0]
                except:
                    continue
            
            # Try looking for heading elements near this input
            try:
                # Look for preceding h1-h6 or strong elements
                heading = input_elem.locator('xpath=preceding::*[self::h1 or self::h2 or self::h3 or self::h4 or self::h5 or self::h6 or self::strong][1]')
                if heading.count() > 0:
                    return heading.first.text_content().strip()
            except:
                pass
                
        except Exception as e:
            logger.error(f"Failed to extract question near input: {e}")
        
        return ""


class AIExamSolver:
    """Main AI-powered exam solver."""
    
    def __init__(self, page, ai: OllamaAI):
        self.page = page
        self.ai = ai
        self.questions = []
        self.answers = {}
        self.correct_answers = {}
    
    def analyze_exam(self, num_questions: int = 10) -> bool:
        """Analyze the exam page and extract all questions."""
        logger.info(f"[*] Analyzing exam structure (expecting {num_questions} questions)...")
        
        self.questions = QuestionExtractor.extract_questions(self.page, num_questions)
        
        if not self.questions:
            logger.error("[!] No questions found on page!")
            return False
        
        logger.info(f"[OK] Found {len(self.questions)} questions")
        return True
    
    def answer_all_questions(self) -> Dict:
        """Use AI to answer all questions."""
        logger.info("\n[AI] Analyzing questions and selecting answers...")
        
        for question in self.questions:
            q_id = question['id']
            q_text = question['question_text']
            options = question['options']
            
            logger.info(f"\n{'='*80}")
            logger.info(f"{q_id}: {q_text}")
            
            # Check if we already know the correct answer
            if q_id in self.correct_answers:
                selected_idx = self.correct_answers[q_id]
                reasoning = "Using previously confirmed correct answer"
                logger.info(f"[KNOWN] Answer: {options[selected_idx]['text']}")
            else:
                # Ask AI
                option_texts = [opt['text'] for opt in options]
                selected_idx, reasoning = self.ai.answer_question(q_text, option_texts)
            
            # Store answer
            selected_option = options[selected_idx]
            self.answers[q_id] = {
                'option_id': selected_option['id'],
                'option_index': selected_idx,
                'option_text': selected_option['text']
            }
            
            logger.info(f"[SELECT] {selected_option['text']}")
            logger.info(f"[REASON] {reasoning}")
        
        return self.answers
    
    def click_answers(self) -> bool:
        """Click the selected answers on the page."""
        logger.info("\n[*] Clicking selected answers...")
        
        try:
            for question in self.questions:
                q_id = question['id']
                
                if q_id not in self.answers:
                    logger.warning(f"[!] No answer stored for {q_id}")
                    continue
                
                answer = self.answers[q_id]
                option_idx = answer['option_index']
                option = question['options'][option_idx]
                
                # Find and click the input
                clicked = self._click_option(question, option)
                
                if clicked:
                    logger.info(f"[OK] {q_id}: Clicked '{option['text'][:30]}...'")
                    time.sleep(0.5)
                else:
                    logger.error(f"[!] {q_id}: Failed to click option")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to click answers: {e}")
            return False
    
    def _click_option(self, question: Dict, option: Dict) -> bool:
        """Click a specific option."""
        try:
            # Get all inputs and click by global index (most reliable)
            all_inputs = self.page.locator("input[type='radio'], input[type='checkbox']")
            global_idx = option.get('global_index')
            
            if global_idx is not None:
                try:
                    input_elem = all_inputs.nth(global_idx)
                    input_elem.scroll_into_view_if_needed()
                    input_elem.click(force=True)
                    return True
                except Exception as e:
                    logger.error(f"Failed to click by global index: {e}")
            
            # Strategy 1: Click by value attribute (UUID)
            if option['input_value']:
                try:
                    selector = f"input[value='{option['input_value']}']"
                    elem = self.page.locator(selector)
                    if elem.count() > 0:
                        elem.first.scroll_into_view_if_needed()
                        elem.first.click(force=True)
                        return True
                except:
                    pass
            
            # Strategy 2: Click by ID
            if option['input_id']:
                try:
                    elem = self.page.locator(f"input#{option['input_id']}")
                    if elem.count() > 0:
                        elem.first.scroll_into_view_if_needed()
                        elem.first.click(force=True)
                        return True
                except:
                    pass
            
            return False
            
        except Exception as e:
            logger.error(f"Error clicking option: {e}")
            return False
    
    def submit_and_check(self) -> Tuple[int, int, bool]:
        """
        Submit exam and check score.
        
        Returns:
            Tuple of (score, total, is_perfect)
        """
        try:
            logger.info("\n[*] Submitting answers...")
            
            # Click submit button
            submit_btn = self.page.get_by_role("button", name="Submit")
            submit_btn.click()
            time.sleep(3)
            
            # Extract score
            score_text = self.page.locator("text=/Your score:/i").inner_text()
            logger.info(f"[SCORE] {score_text}")
            
            match = re.search(r'(\d+)/(\d+)', score_text)
            if match:
                score = int(match.group(1))
                total = int(match.group(2))
                is_perfect = (score == total)
                
                return score, total, is_perfect
            
        except Exception as e:
            logger.error(f"Failed to submit/check score: {e}")
        
        return 0, 0, False
    
    def retest(self, num_questions: int = 10) -> bool:
        """Click retest button - DON'T re-analyze, questions don't change."""
        try:
            logger.info("[*] Starting retest...")
            retest_btn = self.page.get_by_role("button", name="Retest Now")
            retest_btn.click()
            time.sleep(3)
            
            # DON'T re-analyze - questions are already extracted and don't change
            # Just wait for page to be ready
            self.page.wait_for_load_state("networkidle", timeout=10000)
            logger.info("[OK] Retest ready")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to retest: {e}")
            return False


def _print_answer_summary(smart_solver, questions: List[Dict]):
    """Print a JSON summary of correct answers."""
    logger.info("\n" + "="*80)
    logger.info("CORRECT ANSWERS SUMMARY")
    logger.info("="*80)
    
    summary = {}
    
    for question in questions:
        q_id = question['id']
        q_state = smart_solver.state['questions'][q_id]
        
        # Get the confirmed correct answer index
        correct_idx = q_state.get('confirmed_correct_index')
        if correct_idx is None:
            # Use current best guess if not confirmed
            correct_idx = q_state.get('current_answer_index', 0)
        
        # Get the answer text
        correct_answer = question['options'][correct_idx]['text']
        summary[q_id] = {"correct_answer": correct_answer}
    
    # Print as formatted JSON
    print("\n" + json.dumps(summary, indent=2, ensure_ascii=False))
    
    # Also save to file
    try:
        with open("correct_answers.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info(f"\n[SAVED] Answers saved to correct_answers.json")
    except Exception as e:
        logger.error(f"Failed to save answers: {e}")
    
    logger.info("="*80)


def run_ai_solver(page, num_questions: int = 10, max_attempts: int = 50):
    """
    Main function to run AI-powered exam solver with smart brute force.
    
    Args:
        page: Playwright page object
        num_questions: Number of questions in the exam
        max_attempts: Maximum number of attempts
    """
    from smart_solver import SmartBruteForceSolver
    
    # Initialize AI
    logger.info("[*] Initializing AI (Ollama llama3.1:8b)...")
    ai = OllamaAI(model="llama3.1:8b")
    
    # Test AI connection
    test_response = ai.ask("Say 'ready' if you can help solve exam questions.")
    if not test_response:
        logger.error("[!] Cannot connect to Ollama! Make sure it's running: ollama serve")
        return
    logger.info(f"[OK] AI ready: {test_response[:50]}")
    
    # Initialize base solver (for extraction and clicking)
    base_solver = AIExamSolver(page, ai)
    
    # Analyze exam
    if not base_solver.analyze_exam(num_questions):
        logger.error("Failed to analyze exam")
        return
    
    # Initialize smart solver
    smart_solver = SmartBruteForceSolver(ai, base_solver.questions)
    
    # Phase 1: Initial AI attempt
    if smart_solver.state['current_phase'] == 'initial_ai_attempt':
        logger.info("\n" + "="*80)
        logger.info("PHASE 1: AI Initial Reasoning")
        logger.info("="*80)
        
        initial_answers = smart_solver.get_initial_ai_answers()
        
        # Convert to format needed by base_solver
        for q_id, option_idx in initial_answers.items():
            question = next(q for q in base_solver.questions if q['id'] == q_id)
            selected_option = question['options'][option_idx]
            base_solver.answers[q_id] = {
                'option_id': selected_option['id'],
                'option_index': option_idx,
                'option_text': selected_option['text']
            }
        
        # Click and submit
        base_solver.click_answers()
        score, total, is_perfect = base_solver.submit_and_check()
        
        smart_solver.process_result(score, total)
        
        if is_perfect:
            logger.info(f"\n[SUCCESS] Perfect score on first try! {score}/{total}")
            logger.info(smart_solver.get_status_report())
            _print_answer_summary(smart_solver, base_solver.questions)
            return
        
        # Retest for next phase
        base_solver.retest(num_questions)
    
    # Phase 2: Smart brute force
    logger.info("\n" + "="*80)
    logger.info("PHASE 2: Smart Brute Force Cracking")
    logger.info("="*80)
    
    tested_q_id = None
    tested_option = None
    
    for attempt in range(1, max_attempts + 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"[ATTEMPT] {attempt}/{max_attempts}")
        logger.info(smart_solver.get_status_report())
        
        # Get next answers to try
        answers, status = smart_solver.get_next_attempt_answers(
            smart_solver.state['best_score'],
            len(base_solver.questions)
        )
        
        if status == "PERFECT_SCORE":
            logger.info("[SUCCESS] Perfect score achieved!")
            break
        
        if status == "NO_MORE_OPTIONS":
            logger.warning("[WARN] Exhausted all options")
            break
        
        # Parse status to track what we're testing
        if status.startswith("TESTING_"):
            parts = status.split("_")
            tested_q_id = "_".join(parts[1:-2])  # Handle q1, q2, etc.
            tested_option = int(parts[-1]) - 1
            logger.info(f"[TESTING] Changing {tested_q_id} to option {tested_option + 1}")
            logger.info(f"[LOCKED] All other questions use confirmed/best answers")
        
        logger.info(f"{'='*80}")
        
        # Convert to format needed by base_solver
        base_solver.answers = {}
        for q_id, option_idx in answers.items():
            question = next(q for q in base_solver.questions if q['id'] == q_id)
            selected_option = question['options'][option_idx]
            
            # Log which answer we're using for each question
            if q_id == tested_q_id:
                logger.info(f"[NEW] {q_id}: Testing option {option_idx + 1}: {selected_option['text'][:40]}...")
            else:
                q_state = smart_solver.state['questions'][q_id]
                if q_state.get('confirmed_correct_index') is not None:
                    logger.info(f"[LOCKED] {q_id}: Using confirmed option {option_idx + 1}")
                else:
                    logger.info(f"[KEEP] {q_id}: Using option {option_idx + 1}")
            
            base_solver.answers[q_id] = {
                'option_id': selected_option['id'],
                'option_index': option_idx,
                'option_text': selected_option['text']
            }
        
        # Click and submit
        if not base_solver.click_answers():
            logger.error("Failed to click answers")
            break
        
        score, total, is_perfect = base_solver.submit_and_check()
        
        # Process result
        smart_solver.process_result(score, total, tested_q_id, tested_option)
        
        if is_perfect:
            logger.info(f"\n[SUCCESS] PERFECT SCORE! {score}/{total}")
            logger.info(smart_solver.get_status_report())
            break
        
        # Retest for next attempt
        if attempt < max_attempts:
            if not base_solver.retest(num_questions):
                logger.error("Failed to retest")
                break
    
    logger.info(f"\n{'='*80}")
    logger.info("FINAL REPORT")
    logger.info(smart_solver.get_status_report())
    logger.info(f"{'='*80}")
    
    # Print answer summary
    _print_answer_summary(smart_solver, base_solver.questions)