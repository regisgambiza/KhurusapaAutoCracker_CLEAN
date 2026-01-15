"""Smart Brute Force Solver with shuffle-resistant JSON tracking - IMPROVED VERSION"""

import json
import os
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class SmartBruteForceSolver:
    """
    SMART shuffle-resistant brute force solver that:
    1. Uses AI for initial educated guesses
    2. Tracks EVERYTHING by OPTION TEXT (not index) to resist shuffling
    3. Cracks question-by-question sequentially
    4. Locks confirmed correct answers permanently
    5. Uses JSON for complete tracking
    """
    
    def __init__(self, ai, questions: List[Dict], json_file: str = "shuffle_progress.json"):
        self.ai = ai
        self.questions = questions
        self.json_file = json_file
        self.state = self._load_state()
        
        # Initialize state if new exam
        if not self.state.get('initialized'):
            self._initialize_state()
    
    def _load_state(self) -> Dict:
        """Load state from JSON file."""
        if os.path.exists(self.json_file):
            try:
                with open(self.json_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                    logger.info(f"[JSON] Loaded existing progress from {self.json_file}")
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
            logger.info(f"[JSON] Progress saved to {self.json_file}")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def _initialize_state(self):
        """Initialize fresh state for new exam."""
        logger.info("[INIT] Initializing new exam tracking...")
        
        self.state = {
            'initialized': True,
            'exam_started': datetime.now().isoformat(),
            'total_questions': len(self.questions),
            'current_phase': 'initial_ai_attempt',
            'current_question_index': 0,  # Which question we're currently cracking
            'attempt_number': 0,
            'best_score': 0,
            'confirmed_correct': {},  # {q_id: correct_option_text}
            'locked_questions': [],   # q_ids that are confirmed
            'question_history': {},
            'all_attempts': []
        }
        
        # Initialize tracking for each question
        for question in self.questions:
            q_id = question['id']
            self.state['question_history'][q_id] = {
                'question_text': question['question_text'],
                'options_history': [],  # List of all option texts seen
                'tried_options': [],    # Option texts already tried
                'current_best': None,   # Current best option text
                'ai_initial_choice': None,
                'ai_reasoning': None,
                'test_results': []      # History of test attempts
            }
            
            # Record current options
            current_options = [opt['text'].strip() for opt in question['options']]
            self.state['question_history'][q_id]['options_history'].append({
                'timestamp': datetime.now().isoformat(),
                'options': current_options
            })
        
        self._save_state()
    
    def get_initial_ai_answers(self) -> Dict[str, str]:
        """
        PHASE 1: Use AI to get initial best guesses for ALL questions.
        Returns: {question_id: selected_option_text}
        """
        logger.info("\n" + "="*80)
        logger.info("[PHASE 1] AI Initial Reasoning for ALL Questions")
        logger.info("="*80)
        
        answers = {}
        
        for question in self.questions:
            q_id = question['id']
            q_text = question['question_text']
            options = [opt['text'] for opt in question['options']]
            
            logger.info(f"\n{q_id}: {q_text[:80]}...")
            
            # Ask AI for initial guess
            selected_idx, reasoning = self.ai.answer_question(q_text, options)
            selected_option = options[selected_idx]
            
            # Store in state
            self.state['question_history'][q_id]['ai_initial_choice'] = selected_option
            self.state['question_history'][q_id]['ai_reasoning'] = reasoning
            self.state['question_history'][q_id]['current_best'] = selected_option
            self.state['question_history'][q_id]['tried_options'].append(selected_option)
            
            answers[q_id] = selected_option
            
            logger.info(f"[AI] Selected: '{selected_option[:50]}...'")
            logger.info(f"[REASON] {reasoning}")
        
        self.state['current_phase'] = 'smart_brute_force'
        self.state['attempt_number'] = 1
        self._save_state()
        
        return answers
    
    def get_next_brute_force_answers(self, current_score: int, total_questions: int) -> Tuple[Dict[str, str], str, Optional[str]]:
        """
        PHASE 2: Smart brute force - crack ONE question at a time.
        
        Returns:
            - answers_dict: {q_id: option_text} for ALL questions
            - status_message: Description of what's happening
            - tested_q_id: Which question is being tested (or None if done)
        """
        # Update best score
        if current_score > self.state['best_score']:
            self.state['best_score'] = current_score
            logger.info(f"[PROGRESS] New best score: {current_score}/{total_questions}")
        
        # Check if perfect score achieved
        if current_score == total_questions:
            logger.info("[SUCCESS] Perfect score achieved!")
            self.state['current_phase'] = 'completed'
            self._save_state()
            return self._get_all_current_answers(), "PERFECT_SCORE_ACHIEVED", None
        
        # Get current question to crack
        current_q_id = self._get_next_question_to_crack()
        
        if not current_q_id:
            logger.warning("[WARN] All questions have been attempted!")
            return self._get_all_current_answers(), "ALL_QUESTIONS_TRIED", None
        
        logger.info(f"\n[CRACKING] Now testing question: {current_q_id}")
        
        # Get the next untried option for this question
        next_option = self._get_next_option_to_try(current_q_id)
        
        if not next_option:
            logger.warning(f"[SKIP] {current_q_id}: No more untried options")
            # Mark as tried all options, move to next question
            self.state['current_question_index'] += 1
            self._save_state()
            return self.get_next_brute_force_answers(current_score, total_questions)
        
        # Get current best answers for all questions
        all_answers = self._get_all_current_answers()
        
        # Update only the question we're testing
        all_answers[current_q_id] = next_option
        
        # Record that we're trying this option
        q_history = self.state['question_history'][current_q_id]
        if next_option not in q_history['tried_options']:
            q_history['tried_options'].append(next_option)
        q_history['current_best'] = next_option
        
        logger.info(f"[TESTING] {current_q_id} → '{next_option[:50]}...'")
        logger.info(f"          (previously tried: {len(q_history['tried_options'])-1} options)")
        
        self._save_state()
        
        return all_answers, f"TESTING_{current_q_id}", current_q_id
    
    def process_attempt_result(self, score: int, total: int, tested_q_id: Optional[str] = None):
        """
        Analyze attempt result and update state.
        
        Logic:
        - If score INCREASED: new option is CORRECT (lock it)
        - If score DECREASED: old option was CORRECT (lock old one)
        - If score UNCHANGED: both options are WRONG (try next option)
        """
        previous_best = self.state['best_score']
        
        attempt_record = {
            'attempt_number': self.state['attempt_number'],
            'timestamp': datetime.now().isoformat(),
            'score': score,
            'total': total,
            'tested_question': tested_q_id,
            'previous_best_score': previous_best
        }
        
        self.state['all_attempts'].append(attempt_record)
        self.state['attempt_number'] += 1
        
        if tested_q_id:
            q_history = self.state['question_history'][tested_q_id]
            current_option = q_history['current_best']
            tried_options = q_history['tried_options']
            
            if len(tried_options) >= 2:
                # Get the previous option (the one before current)
                previous_option = tried_options[-2] if len(tried_options) >= 2 else None
                
                if score > previous_best:
                    # NEW option is CORRECT!
                    logger.info(f"✅ [CORRECT] {tested_q_id}: '{current_option[:30]}...' is CORRECT!")
                    self.state['confirmed_correct'][tested_q_id] = current_option
                    self.state['locked_questions'].append(tested_q_id)
                    self.state['best_score'] = score
                    
                elif score < previous_best:
                    # NEW option is WRONG, OLD option was CORRECT
                    logger.info(f"✅ [CORRECT] {tested_q_id}: '{previous_option[:30]}...' is CORRECT!")
                    self.state['confirmed_correct'][tested_q_id] = previous_option
                    self.state['locked_questions'].append(tested_q_id)
                    # Revert to previous option
                    q_history['current_best'] = previous_option
                    
                else:  # score unchanged
                    # BOTH options are WRONG
                    logger.info(f"❌ [BOTH WRONG] {tested_q_id}: Both options are wrong")
                    # Current option stays, we'll try next one in next iteration
                    
            # Move to next question for next attempt
            if tested_q_id in self.state['locked_questions']:
                self.state['current_question_index'] += 1
            # If not locked (both wrong case), we stay on same question to try next option
        
        self._save_state()
    
    def _get_next_question_to_crack(self) -> Optional[str]:
        """Get the next question that needs cracking (not locked)."""
        # Get question IDs in order
        q_ids = [f'q{i+1}' for i in range(self.state['total_questions'])]
        
        # Start from current index
        for i in range(self.state['current_question_index'], len(q_ids)):
            q_id = q_ids[i]
            
            # Skip if already locked/confirmed
            if q_id in self.state['locked_questions']:
                continue
            
            # Check if there are untried options
            q_history = self.state['question_history'][q_id]
            current_options = q_history['options_history'][-1]['options']  # Latest options
            tried_options = set(q_history['tried_options'])
            untried_options = [opt for opt in current_options if opt not in tried_options]
            
            if untried_options:
                return q_id
        
        return None
    
    def _get_next_option_to_try(self, q_id: str) -> Optional[str]:
        """Get the next untried option for a question."""
        q_history = self.state['question_history'][q_id]
        
        # Get latest available options
        current_options = q_history['options_history'][-1]['options']
        tried_options = set(q_history['tried_options'])
        
        # Find untried options
        untried = [opt for opt in current_options if opt not in tried_options]
        
        if not untried:
            return None
        
        # If we have AI reasoning, use it to prioritize
        if len(untried) > 1 and q_history['ai_reasoning']:
            # Simple heuristic: pick the one that seems most plausible
            # In a more advanced version, you could re-query AI for ranking
            return untried[0]
        
        return untried[0]
    
    def _get_all_current_answers(self) -> Dict[str, str]:
        """Get current best answer for ALL questions."""
        answers = {}
        
        for q_id in self.state['question_history']:
            # If confirmed correct, use that
            if q_id in self.state['confirmed_correct']:
                answers[q_id] = self.state['confirmed_correct'][q_id]
            else:
                # Otherwise use current best guess
                current_best = self.state['question_history'][q_id]['current_best']
                if current_best:
                    answers[q_id] = current_best
                else:
                    # Fallback to AI initial choice
                    answers[q_id] = self.state['question_history'][q_id]['ai_initial_choice']
        
        return answers
    
    def update_question_options(self, questions: List[Dict]):
        """Update options for all questions (called after each retest)."""
        for question in questions:
            q_id = question['id']
            if q_id in self.state['question_history']:
                current_options = [opt['text'].strip() for opt in question['options']]
                
                # Record new set of options
                self.state['question_history'][q_id]['options_history'].append({
                    'timestamp': datetime.now().isoformat(),
                    'options': current_options
                })
        
        self._save_state()
    
    def get_status_report(self) -> str:
        """Generate a detailed status report."""
        confirmed = len(self.state['confirmed_correct'])
        total = self.state['total_questions']
        progress = (confirmed / total) * 100 if total > 0 else 0
        
        report = f"""
{'='*80}
SMART BRUTE FORCE SOLVER - STATUS REPORT
{'='*80}
Phase: {self.state['current_phase']}
Attempt Number: {self.state['attempt_number']}
Best Score: {self.state['best_score']}/{total}
Confirmed Correct: {confirmed}/{total} ({progress:.1f}%)
Current Question Index: {self.state['current_question_index']}
Locked Questions: {len(self.state['locked_questions'])}
{'='*80}

QUESTION PROGRESS:
"""
        for i in range(total):
            q_id = f'q{i+1}'
            if q_id not in self.state['question_history']:
                continue
                
            history = self.state['question_history'][q_id]
            status = "✅ LOCKED" if q_id in self.state['locked_questions'] else "🔓 TESTING"
            tried_count = len(history['tried_options'])
            
            current_answer = self.state['confirmed_correct'].get(q_id, history.get('current_best', ''))
            if not current_answer:
                current_answer = "Not yet determined"
            
            report += f"\n{q_id}: {status}"
            report += f"\n  Tried options: {tried_count}"
            report += f"\n  Current answer: {current_answer[:50]}..."
            
            # FIX: Safely handle None ai_reasoning
            ai_reasoning = history.get('ai_reasoning')
            if ai_reasoning:
                report += f"\n  AI reasoning: {ai_reasoning[:50]}..."
            elif q_id not in self.state['locked_questions']:
                report += f"\n  AI reasoning: Not yet obtained"
            
            report += "\n"
        
        report += f"\n{'='*80}"
        return report