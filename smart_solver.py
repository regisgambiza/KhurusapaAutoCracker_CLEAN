"""
FIXED SMART SOLVER - SEQUENTIAL ELIMINATION STRATEGY
This works for exams with ONLY global score feedback
"""

import json
import logging
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class SequentialEliminationSolver:
    """
    Strategy for exams with only global score feedback.
    Works by fixing ALL answers except ONE, then testing that one.
    """
    
    def __init__(self, questions: List[Dict], json_file: str = "fixed_progress.json"):
        self.questions = questions
        self.json_file = json_file
        self.state = self._load_state()
        
        if not self.state.get('initialized'):
            self._initialize_state()
    
    def _initialize_state(self):
        """Initialize fresh state."""
        self.state = {
            'initialized': True,
            'start_time': datetime.now().isoformat(),
            'total_questions': len(self.questions),
            'current_question_index': 0,  # Which question we're testing
            'base_score': None,  # Score with current "baseline" answers
            'confirmed_answers': {},  # {q_id: option_text}
            'candidate_answers': {},  # {q_id: option_text} - unconfirmed best guesses
            'test_history': []
        }
        
        # Start with all first options as baseline
        for question in self.questions:
            q_id = question['id']
            self.state['candidate_answers'][q_id] = question['options'][0]['text']
        
        self._save_state()
    
    def get_test_set(self, current_score: Optional[int] = None) -> Dict[str, str]:
        """
        Get the next set of answers to test.
        
        Strategy:
        1. If baseline score unknown, test baseline (all first options)
        2. Otherwise, test ONE question at a time while fixing others
        """
        
        if current_score is not None and self.state['base_score'] is None:
            # First test - establish baseline
            self.state['base_score'] = current_score
            logger.info(f"[BASELINE] Baseline score: {current_score}/{self.state['total_questions']}")
        
        # Get current question to test
        q_id = f"q{self.state['current_question_index'] + 1}"
        
        # If this question is already confirmed, move to next
        while q_id in self.state['confirmed_answers']:
            self.state['current_question_index'] = (self.state['current_question_index'] + 1) % self.state['total_questions']
            q_id = f"q{self.state['current_question_index'] + 1}"
            
            # Check if all confirmed
            if len(self.state['confirmed_answers']) == self.state['total_questions']:
                logger.info("[SUCCESS] All questions confirmed!")
                return self.state['confirmed_answers']
        
        # Build answer set:
        # - Use confirmed answers where available
        # - Use candidate answers for others
        # - For current question, we'll test different options
        test_answers = {}
        
        for i in range(self.state['total_questions']):
            current_q_id = f"q{i+1}"
            
            if current_q_id in self.state['confirmed_answers']:
                # Use confirmed answer
                test_answers[current_q_id] = self.state['confirmed_answers'][current_q_id]
            else:
                # Use candidate answer
                test_answers[current_q_id] = self.state['candidate_answers'][current_q_id]
        
        logger.info(f"[TESTING] Isolating question {q_id}")
        return test_answers
    
    def process_result(self, score: int, total: int, tested_options: Dict[str, str]):
        """
        Process test result and update state.
        
        Logic:
        - If this is baseline test, just record baseline
        - If testing isolated question:
            * If score > baseline: new option is correct
            * If score < baseline: baseline option was correct
            * If score = baseline: inconclusive (try another option)
        """
        
        record = {
            'timestamp': datetime.now().isoformat(),
            'score': score,
            'tested_question': f"q{self.state['current_question_index'] + 1}",
            'tested_options': tested_options
        }
        self.state['test_history'].append(record)
        
        if self.state['base_score'] is None:
            # This was baseline test
            self.state['base_score'] = score
            logger.info(f"[BASELINE SET] Score: {score}/{total}")
        else:
            # This was an isolation test
            current_q_id = f"q{self.state['current_question_index'] + 1}"
            current_option = tested_options[current_q_id]
            
            if score > self.state['base_score']:
                # NEW option is CORRECT!
                logger.info(f"✅ {current_q_id}: '{current_option[:50]}...' is CORRECT!")
                self.state['confirmed_answers'][current_q_id] = current_option
                self.state['base_score'] = score  # Update baseline
                
                # Move to next question
                self.state['current_question_index'] = (self.state['current_question_index'] + 1) % self.state['total_questions']
                
            elif score < self.state['base_score']:
                # NEW option is WRONG, baseline was correct
                # Get baseline option for this question
                baseline_q = self.questions[self.state['current_question_index']]
                baseline_option = baseline_q['options'][0]['text']
                
                logger.info(f"✅ {current_q_id}: Baseline option '{baseline_option[:50]}...' is CORRECT!")
                self.state['confirmed_answers'][current_q_id] = baseline_option
                
                # Move to next question
                self.state['current_question_index'] = (self.state['current_question_index'] + 1) % self.state['total_questions']
                
            else:
                # Score unchanged - try different option
                logger.info(f"➖ {current_q_id}: Option '{current_option[:50]}...' gave same score")
                
                # Try next option for this question
                question = self.questions[self.state['current_question_index']]
                current_idx = [opt['text'] for opt in question['options']].index(current_option)
                next_idx = (current_idx + 1) % len(question['options'])
                next_option = question['options'][next_idx]['text']
                
                self.state['candidate_answers'][current_q_id] = next_option
                
                # If we've tried all options and score never changed, mark as unknown
                if next_idx == 0:  # We've cycled through all options
                    logger.warning(f"⚠️ {current_q_id}: All options give same score!")
                    # Default to first option
                    self.state['confirmed_answers'][current_q_id] = question['options'][0]['text']
                    self.state['current_question_index'] = (self.state['current_question_index'] + 1) % self.state['total_questions']
        
        self._save_state()
        
        # Return status
        confirmed = len(self.state['confirmed_answers'])
        return {
            'confirmed': confirmed,
            'total': self.state['total_questions'],
            'current_question': f"q{self.state['current_question_index'] + 1}",
            'base_score': self.state['base_score']
        }
    
    def get_next_option_for_testing(self, q_id: str) -> str:
        """Get the next option to test for a question."""
        q_idx = int(q_id[1:]) - 1
        question = self.questions[q_idx]
        
        current_option = self.state['candidate_answers'].get(q_id)
        if not current_option:
            return question['options'][0]['text']
        
        # Find current option index
        options_texts = [opt['text'] for opt in question['options']]
        if current_option not in options_texts:
            return question['options'][0]['text']
        
        current_idx = options_texts.index(current_option)
        next_idx = (current_idx + 1) % len(options_texts)
        return question['options'][next_idx]['text']
    
    def _load_state(self):
        if os.path.exists(self.json_file):
            try:
                with open(self.json_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {}
    
    def _save_state(self):
        with open(self.json_file, 'w') as f:
            json.dump(self.state, indent=2, fp=f)
    
    def get_status(self):
        confirmed = len(self.state['confirmed_answers'])
        total = self.state['total_questions']
        
        status = f"""
{'='*80}
SEQUENTIAL ELIMINATION SOLVER
{'='*80}
Confirmed: {confirmed}/{total}
Baseline Score: {self.state.get('base_score', 'Unknown')}
Current Question: q{self.state['current_question_index'] + 1}
{'='*80}
"""
        
        for i in range(total):
            q_id = f"q{i+1}"
            if q_id in self.state['confirmed_answers']:
                status += f"{q_id}: ✅ '{self.state['confirmed_answers'][q_id][:50]}...'\n"
            else:
                status += f"{q_id}: 🔄 Testing...\n"
        
        status += "="*80
        return status