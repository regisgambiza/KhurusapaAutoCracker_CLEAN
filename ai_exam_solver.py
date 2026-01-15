"""AI-Powered Exam Solver using Ollama llama3.1:8b - IMPROVED VERSION"""

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
        
        try:
            all_inputs = page.locator("input[type='radio'], input[type='checkbox']")
            all_labels = page.locator("label")
            total_inputs = all_inputs.count()
            total_labels = all_labels.count()
            
            logger.info(f"[DEBUG] Found {total_inputs} input elements and {total_labels} labels")
            
            if total_inputs == 0:
                logger.error("[!] No radio/checkbox inputs found on page!")
                return []
            
            options_per_question = total_inputs // num_questions
            
            for q_idx in range(num_questions):
                start_idx = q_idx * options_per_question
                end_idx = start_idx + options_per_question if q_idx < num_questions - 1 else total_inputs
                
                first_input = all_inputs.nth(start_idx)
                question_text = QuestionExtractor._extract_question_near_input(first_input, page) or f"Question {q_idx + 1}"
                
                options = []
                for opt_idx, global_idx in enumerate(range(start_idx, end_idx)):
                    label_elem = all_labels.nth(global_idx) if global_idx < total_labels else None
                    option_text = label_elem.text_content().strip() if label_elem else f"Option {opt_idx + 1}"
                    
                    try:
                        input_elem = all_inputs.nth(global_idx)
                        opt_data = {
                            'text': option_text,
                            'input_id': input_elem.get_attribute('id'),
                            'input_name': input_elem.get_attribute('name'),
                            'input_value': input_elem.get_attribute('value')
                        }
                    except:
                        opt_data = {'text': option_text}
                    
                    options.append(opt_data)
                
                questions.append({
                    'id': f'q{q_idx + 1}',
                    'question_text': question_text,
                    'options': options
                })
                
                logger.info(f"[OK] Extracted Q{q_idx + 1}: '{question_text[:50]}...' ({len(options)} options)")
            
            return questions
            
        except Exception as e:
            logger.error(f"Failed to extract questions: {e}")
            return []
    
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
    def __init__(self, page, ai: OllamaAI):
        self.page = page
        self.ai = ai
        self.questions: List[Dict] = []
        self.answers: Dict[str, Dict] = {}
        self.known_correct: Dict[str, str] = {}

    def analyze_exam(self, num_questions: int = 10) -> bool:
        logger.info(f"[*] Extracting questions (expecting {num_questions})...")
        self.questions = QuestionExtractor.extract_questions(self.page, num_questions)
        if not self.questions:
            logger.error("Failed to extract any questions!")
            return False
        logger.info(f"[OK] Extracted {len(self.questions)} questions")
        return True

    def answer_all_questions(self):
        logger.info("\n[AI] Selecting answers...")
        for q in self.questions:
            qid = q['id']
            qtext = q['question_text']
            option_texts = [o['text'] for o in q['options']]

            if qid in self.known_correct:
                selected_text = self.known_correct[qid]
                reason = "Previously confirmed correct"
            else:
                idx, reason = self.ai.answer_question(qtext, option_texts)
                selected_text = option_texts[idx]

            self.answers[qid] = {
                'option_text': selected_text,
                'reasoning': reason
            }

            logger.info(f"  {qid} → {selected_text[:70]}...")
            logger.debug(f"     {reason}")

    def click_answers(self) -> bool:
        """Enhanced version with special handling for 'All the answers are correct'"""
        logger.info("\n[*] Clicking answers by TEXT (shuffle resistant)...")
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
                if self._click_all_answers_correct(q, target_text):
                    success_count += 1
                    time.sleep(1.0)
                    continue
                else:
                    logger.error(f"Failed to click 'All answers correct' for {qid}")
                    continue

            # Regular clicking for other answers
            if self._click_by_text(target_text, question_text):
                success_count += 1
                time.sleep(1.0)
            else:
                logger.error(f"Failed to click answer for {qid}: {target_text[:60]}...")

        logger.info(f"Successfully clicked {success_count}/{len(self.questions)} answers")
        return success_count == len(self.questions)

    def _click_all_answers_correct(self, question: Dict, target_text: str) -> bool:
        """
        Special handler ONLY for 'All the answers are correct' option.
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
            logger.error(f"Error in _click_all_answers_correct: {e}")
            return False

    def _click_by_text(self, target_text: str, question_text: str = "") -> bool:
        """
        Enhanced clicking with multiple fallback strategies.
        Handles ambiguous phrases like "All the answers are correct"
        """
        try:
            target_text = ' '.join(target_text.split()).strip()
            
            # Strategy 1: Try exact match first (most reliable)
            logger.debug(f"Strategy 1: Exact match for '{target_text}'")
            exact_labels = self.page.locator("label").filter(has_text=re.compile(f"^{re.escape(target_text)}$"))
            if exact_labels.count() > 0:
                try:
                    exact_labels.first.click(force=True, timeout=10000)
                    logger.info(f"✓ Clicked using exact match")
                    return True
                except:
                    pass

            # Strategy 2: Scope to question container if we have question text
            if question_text:
                logger.debug(f"Strategy 2: Searching within question container")
                question_container = self._find_question_container(question_text)
                
                if question_container:
                    # Try exact match within container
                    scoped_labels = question_container.locator("label").filter(has_text=target_text)
                    if scoped_labels.count() > 0:
                        try:
                            scoped_labels.first.click(force=True, timeout=10000)
                            logger.info(f"✓ Clicked within question container (exact)")
                            return True
                        except:
                            pass
                    
                    # Try partial match within container (safer when scoped)
                    if len(target_text) > 20:
                        short_text = target_text[:30]
                        scoped_partial = question_container.locator("label").filter(has_text=short_text)
                        if scoped_partial.count() > 0:
                            try:
                                scoped_partial.first.click(force=True, timeout=10000)
                                logger.info(f"✓ Clicked within container (partial: '{short_text}')")
                                return True
                            except:
                                pass

            # Strategy 3: Handle special cases (common ambiguous phrases)
            logger.debug(f"Strategy 3: Special case handling")
            if self._click_special_case(target_text, question_text):
                return True

            # Strategy 4: Progressive partial matching (from long to short)
            logger.debug(f"Strategy 4: Progressive partial matching")
            lengths = [50, 35, 25, 20] if len(target_text) > 50 else [35, 25, 20]
            
            for length in lengths:
                if len(target_text) <= length:
                    continue
                    
                partial = target_text[:length].strip()
                logger.debug(f"  Trying partial: '{partial}'")
                
                labels = self.page.locator("label").filter(has_text=partial)
                if labels.count() > 0:
                    try:
                        # If multiple matches, try to pick the right one
                        if labels.count() == 1:
                            labels.first.click(force=True, timeout=10000)
                            logger.info(f"✓ Clicked using partial ({length} chars)")
                            return True
                        else:
                            # Multiple matches - try to find best one
                            for i in range(min(3, labels.count())):
                                label = labels.nth(i)
                                full_text = label.text_content().strip()
                                if target_text in full_text:
                                    label.click(force=True, timeout=10000)
                                    logger.info(f"✓ Clicked best match from {labels.count()} options")
                                    return True
                    except Exception as e:
                        logger.debug(f"  Partial match failed: {e}")
                        continue

            # Strategy 5: Keyword-based fallback (last resort)
            logger.debug(f"Strategy 5: Keyword fallback")
            keywords = self._extract_keywords(target_text)
            for keyword in keywords:
                if len(keyword) < 5:  # Skip very short keywords
                    continue
                    
                logger.debug(f"  Trying keyword: '{keyword}'")
                kw_labels = self.page.locator("label").filter(has_text=keyword)
                
                if kw_labels.count() > 0:
                    try:
                        # Verify it's the right match
                        for i in range(min(3, kw_labels.count())):
                            label = kw_labels.nth(i)
                            full_text = label.text_content().strip()
                            # Check if more than one keyword matches (higher confidence)
                            match_count = sum(1 for kw in keywords if kw.lower() in full_text.lower())
                            if match_count >= 2:
                                label.click(force=True, timeout=10000)
                                logger.warning(f"⚠ Clicked using keyword match: '{keyword}' (low confidence)")
                                return True
                    except:
                        continue

            logger.error(f"✗ All strategies failed for: '{target_text}'")
            return False

        except Exception as e:
            logger.error(f"Click error: {e}")
            return False

    def _find_question_container(self, question_text: str):
        """Find the container div/fieldset for a specific question"""
        try:
            # Clean question text for searching
            search_text = question_text[:50].replace("'", "\\'").replace('"', '\\"')
            
            # Try various container patterns
            patterns = [
                f"xpath=//*[contains(text(), '{search_text}')]/ancestor::div[contains(@class, 'question')][1]",
                f"xpath=//*[contains(text(), '{search_text}')]/ancestor::fieldset[1]",
                f"xpath=//*[contains(text(), '{search_text}')]/ancestor::div[.//input[@type='radio' or @type='checkbox']][1]",
                f"xpath=//*[contains(text(), '{search_text}')]/ancestor::form[1]",
            ]
            
            for pattern in patterns:
                try:
                    container = self.page.locator(pattern).first
                    if container.count() > 0:
                        logger.debug(f"  Found container using: {pattern.split('/')[0]}")
                        return container
                except:
                    continue
                    
        except Exception as e:
            logger.debug(f"Container search error: {e}")
        
        return None

    def _click_special_case(self, target_text: str, question_text: str = "") -> bool:
        """Handle special ambiguous cases"""
        
        # Case 1: "All the answers are correct"
        if "all" in target_text.lower() and "answer" in target_text.lower() and "correct" in target_text.lower():
            logger.debug("  Special case: 'All answers correct' variant")
            
            # Try multiple variations
            variations = [
                target_text,  # Full text
                "All the answers are correct",
                "All the answers are correct.",
                "All of the answers are correct",
                "All answers are correct",
            ]
            
            for variant in variations:
                labels = self.page.locator("label").filter(has_text=re.compile(re.escape(variant), re.IGNORECASE))
                if labels.count() > 0:
                    try:
                        labels.first.click(force=True, timeout=10000)
                        logger.info(f"✓ Clicked special case variant: '{variant}'")
                        return True
                    except:
                        continue
            
            # Try finding by unique ending if within question context
            if question_text:
                container = self._find_question_container(question_text)
                if container:
                    all_labels = container.locator("label")
                    for i in range(all_labels.count()):
                        label_text = all_labels.nth(i).text_content().strip().lower()
                        if "all" in label_text and "answer" in label_text and "correct" in label_text:
                            try:
                                all_labels.nth(i).click(force=True, timeout=10000)
                                logger.info(f"✓ Clicked 'all answers' within container")
                                return True
                            except:
                                continue
        
        return False

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text for fallback matching"""
        # Remove common words
        common_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                        'could', 'should', 'may', 'might', 'can', 'to', 'of', 'in',
                        'for', 'on', 'at', 'by', 'with', 'from'}
        
        words = text.split()
        keywords = []
        
        # Extract significant words (longer than 4 chars, not common)
        for word in words:
            clean_word = word.strip('.,!?;:()[]{}\"\'').lower()
            if len(clean_word) > 4 and clean_word not in common_words:
                keywords.append(clean_word)
        
        # Prioritize longer words and proper nouns
        keywords.sort(key=lambda x: len(x), reverse=True)
        
        return keywords[:5]  # Return top 5 keywords

    def submit_and_check(self) -> Tuple[int, int, bool]:
        try:
            logger.info("Submitting...")

            submit_btn = self.page.get_by_role(
                "button",
                name=re.compile(r"submit|finish|send|complete", re.I)
            )
            submit_btn.click()
            time.sleep(3.5)

            # ✅ FIX: narrow to elements that actually contain numbers like 5/10
            score_locator = self.page.locator(
                "text=/\\d+\\s*\\/\\s*\\d+/"
            ).filter(
                has_text=re.compile(r"score|result|mark|grade", re.I)
            )

            # If multiple, pick the one that actually contains X/Y
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
        """Very important: re-extract questions every time because order usually changes"""
        try:
            logger.info("Retesting → re-extracting all questions...")
            retest_btn = self.page.get_by_role("button", name=re.compile(r"retest|try again|redo|restart|again", re.I))
            retest_btn.click()
            time.sleep(2.5)

            self.page.wait_for_load_state("networkidle", timeout=15000)
            
            # Re-analyze = re-extract fresh positions & text-order
            return self.analyze_exam(num_questions)
            
        except Exception as e:
            logger.error(f"Retest failed: {e}")
            return False


def run_ai_solver(page, num_questions: int = 10, max_attempts: int = 30):
    ai = OllamaAI()
    solver = AIExamSolver(page, ai)

    if not solver.analyze_exam(num_questions):
        return

    attempt = 0
    while attempt < max_attempts:
        attempt += 1
        logger.info(f"\n===== ATTEMPT {attempt} =====")

        solver.answer_all_questions()
        if not solver.click_answers():
            logger.warning("Some answers could not be clicked - may fail")

        score, total, is_perfect = solver.submit_and_check()
        logger.info(f"Score: {score}/{total}")

        if is_perfect:
            logger.info("PERFECT SCORE ACHIEVED!")
            break

        if attempt >= max_attempts:
            logger.info("Max attempts reached.")
            break

        if not solver.retest(num_questions):
            logger.error("Cannot continue - retest failed")
            break

    logger.info("Finished.")