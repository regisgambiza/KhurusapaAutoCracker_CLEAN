"""Smart Brute Force Solver with AI reasoning and JSON tracking"""

import json
import os
import logging
from typing import Dict, List, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


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
                'ai_reasoning': None
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
            options = question['options']
            
            logger.info(f"\n{q_id}: {q_text}")
            
            # Ask AI
            option_texts = [opt['text'] for opt in options]
            selected_idx, reasoning = self.ai.answer_question(q_text, option_texts)
            
            # Store in state
            self.state['questions'][q_id]['current_answer_index'] = selected_idx
            self.state['questions'][q_id]['tried_indices'].append(selected_idx)
            self.state['questions'][q_id]['ai_reasoning'] = reasoning
            
            answers[q_id] = selected_idx
            
            logger.info(f"[AI] Selected option {selected_idx + 1}: {options[selected_idx]['text'][:50]}...")
            logger.info(f"[REASON] {reasoning}")
        
        self.state['current_phase'] = 'smart_brute_force'
        self._save_state()
        
        return answers
    
    def get_next_attempt_answers(self, last_score: int, last_total: int) -> Tuple[Dict[str, int], str]:
        """
        Phase 2: Smart brute force - crack question by question.
        
        Args:
            last_score: Score from last attempt
            last_total: Total questions
            
        Returns:
            Tuple of (answers_dict, status_message)
        """
        # Update best score
        if last_score > self.state['best_score']:
            self.state['best_score'] = last_score
            logger.info(f"[PROGRESS] New best score: {last_score}/{last_total}")
        
        # Check if perfect
        if last_score == last_total:
            logger.info(f"[SUCCESS] Perfect score achieved!")
            self.state['current_phase'] = 'completed'
            self._save_state()
            return {}, "PERFECT_SCORE"
        
        # Find next question to crack
        question_to_crack = self._find_next_question_to_crack()
        
        if not question_to_crack:
            logger.warning("[WARN] No more questions to crack!")
            return {}, "NO_MORE_OPTIONS"
        
        q_id = question_to_crack['id']
        q_state = self.state['questions'][q_id]
        
        # Get next best option using AI
        next_option_idx = self._get_next_best_option(question_to_crack)
        
        if next_option_idx is None:
            logger.warning(f"[WARN] {q_id}: All options exhausted")
            # Mark as tried all options
            q_state['locked'] = True
            self._save_state()
            return self._get_current_best_answers(), "CONTINUE"
        
        logger.info(f"\n[CRACK] Testing {q_id} with option {next_option_idx + 1}")
        
        # Update state for this question
        old_idx = q_state['current_answer_index']
        q_state['current_answer_index'] = next_option_idx
        if next_option_idx not in q_state['tried_indices']:
            q_state['tried_indices'].append(next_option_idx)
        
        # Build answer set: new option for cracking question, locked answers for others
        answers = self._get_current_best_answers()
        answers[q_id] = next_option_idx
        
        self._save_state()
        
        status = f"TESTING_{q_id}_OPT_{next_option_idx + 1}"
        return answers, status
    
    def process_result(self, score: int, total: int, tested_q_id: str = None, tested_option_idx: int = None):
        """
        Process the result of an attempt and update state.
        
        Args:
            score: Score received
            total: Total questions
            tested_q_id: Question ID that was tested (if any)
            tested_option_idx: Option index that was tested
        """
        attempt_record = {
            'attempt_number': self.state['attempt_number'],
            'timestamp': datetime.now().isoformat(),
            'score': score,
            'total': total,
            'tested_question': tested_q_id,
            'tested_option': tested_option_idx,
            'answers': {q_id: q_state['current_answer_index'] 
                       for q_id, q_state in self.state['questions'].items()}
        }
        
        self.state['attempt_history'].append(attempt_record)
        
        # If we were testing a specific question
        if tested_q_id and tested_option_idx is not None:
            prev_score = self.state['best_score']
            q_state = self.state['questions'][tested_q_id]
            
            if score > prev_score:
                # New option is CORRECT!
                logger.info(f"[CORRECT] {tested_q_id} option {tested_option_idx + 1} is CORRECT!")
                q_state['confirmed_correct_index'] = tested_option_idx
                q_state['locked'] = True
                self.state['best_score'] = score
                
            elif score < prev_score:
                # New option is WRONG, revert to previous
                logger.info(f"[WRONG] {tested_q_id} option {tested_option_idx + 1} is wrong, reverting")
                old_idx = self._get_previous_best_option(tested_q_id)
                q_state['current_answer_index'] = old_idx
                # Lock the old one as correct
                q_state['confirmed_correct_index'] = old_idx
                q_state['locked'] = True
                
            else:
                # Score unchanged - both options are wrong
                logger.info(f"[BOTH_WRONG] {tested_q_id}: Both options {tested_option_idx + 1} and previous are wrong")
                # Don't lock, try next option
        
        self.state['attempt_number'] += 1
        self._save_state()
    
    def _find_next_question_to_crack(self) -> Dict:
        """Find the next question that needs cracking."""
        for question in self.questions:
            q_id = question['id']
            q_state = self.state['questions'][q_id]
            
            # Skip locked questions (already confirmed correct)
            if q_state['locked'] and q_state['confirmed_correct_index'] is not None:
                continue
            
            # Skip if all options tried
            if len(q_state['tried_indices']) >= q_state['num_options']:
                continue
            
            return question
        
        return None
    
    def _get_next_best_option(self, question: Dict) -> int:
        """Use AI to suggest next best option to try."""
        q_id = question['id']
        q_state = self.state['questions'][q_id]
        tried_indices = q_state['tried_indices']
        
        # Get untried options
        all_indices = list(range(q_state['num_options']))
        untried = [i for i in all_indices if i not in tried_indices]
        
        if not untried:
            return None
        
        # If only one untried option, return it
        if len(untried) == 1:
            return untried[0]
        
        # Ask AI to rank remaining options
        q_text = question['question_text']
        untried_options = [question['options'][i]['text'] for i in untried]
        
        prompt = f"""You are helping solve an exam question. You've already tried some options and they were wrong.

Question: {q_text}

Already tried (WRONG): {[question['options'][i]['text'] for i in tried_indices]}

Remaining options:
{chr(10).join([f"{i+1}. {opt}" for i, opt in enumerate(untried_options)])}

Which remaining option is MOST LIKELY correct? Respond with ONLY the number (1, 2, 3, etc.)."""

        response = self.ai.ask(prompt, temperature=0.1)
        
        try:
            # Extract number from response
            import re
            match = re.search(r'\d+', response)
            if match:
                choice = int(match.group()) - 1
                if 0 <= choice < len(untried):
                    return untried[choice]
        except:
            pass
        
        # Fallback: return first untried
        return untried[0]
    
    def _get_previous_best_option(self, q_id: str) -> int:
        """Get the previous option before current one."""
        q_state = self.state['questions'][q_id]
        tried = q_state['tried_indices']
        current = q_state['current_answer_index']
        
        if len(tried) >= 2:
            # Return the one before current
            for i in range(len(tried) - 1, -1, -1):
                if tried[i] != current:
                    return tried[i]
        
        return tried[0] if tried else 0
    
    def _get_current_best_answers(self) -> Dict[str, int]:
        """Get current best answer for each question."""
        answers = {}
        for q_id, q_state in self.state['questions'].items():
            if q_state['confirmed_correct_index'] is not None:
                # Use confirmed correct answer
                answers[q_id] = q_state['confirmed_correct_index']
            else:
                # Use current best guess
                answers[q_id] = q_state['current_answer_index']
        
        return answers
    
    def get_status_report(self) -> str:
        """Generate a status report."""
        locked_count = sum(1 for q in self.state['questions'].values() 
                          if q['locked'] and q['confirmed_correct_index'] is not None)
        
        report = f"""
{'='*80}
SOLVER STATUS REPORT
{'='*80}
Phase: {self.state['current_phase']}
Attempt: {self.state['attempt_number']}
Best Score: {self.state['best_score']}/{self.state['total_questions']}
Confirmed Correct: {locked_count}/{self.state['total_questions']}
{'='*80}
"""
        return report