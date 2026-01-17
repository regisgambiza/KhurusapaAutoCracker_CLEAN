"""AI-Powered Navigation Helper using Ollama for intelligent page state detection"""

import logging
import time
import re
import json
import requests
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PageState:
    """Represents the analyzed state of a page"""
    actual_state: str  # quiz, results, loading, error, login, unknown
    ready_to_proceed: bool
    confidence: int  # 0-100
    reasoning: str
    blocking_issues: List[str]
    recommended_wait_seconds: int


class AINavigationHelper:
    """Use Ollama AI to intelligently verify page states and navigation readiness."""
    
    def __init__(self, model: str = "llama3.1:8b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
    
    def ask_ai(self, prompt: str, temperature: float = 0.1) -> str:
        """Ask Ollama AI a question with low temperature for consistent responses."""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "temperature": temperature
            }
            response = requests.post(self.api_url, json=payload, timeout=30)
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except Exception as e:
            logger.error(f"AI API error: {e}")
            return ""
    
    def get_dom_data(self, page) -> Dict:
        """Extract DOM structure information from the page."""
        try:
            dom_data = {
                'radio_count': page.locator("input[type='radio']").count(),
                'checkbox_count': page.locator("input[type='checkbox']").count(),
                'label_count': page.locator("label").count(),
                'button_count': page.locator("button").count(),
                'loading_indicators': page.locator(".spinner, .loading, [class*='load'], [class*='spin']").count(),
                'modal_overlays': page.locator(".modal, .overlay, [class*='modal'], [class*='overlay']").count(),
                'submit_buttons': self._count_buttons_with_text(page, ["submit", "finish", "send", "complete"]),
                'retest_buttons': self._count_buttons_with_text(page, ["retest", "try again", "redo", "restart"]),
                'input_disabled_count': page.locator("input:disabled").count(),
                'form_count': page.locator("form").count(),
            }
            
            # Check for specific text patterns
            body_text = page.inner_text("body").lower()
            dom_data['has_score_pattern'] = bool(re.search(r'\d+\s*/\s*\d+', page.inner_text("body")))
            dom_data['has_loading_text'] = any(word in body_text for word in ['loading', 'please wait', 'processing'])
            dom_data['has_error_text'] = any(word in body_text for word in ['error', 'failed', 'try again', 'something went wrong'])
            
            return dom_data
        except Exception as e:
            logger.error(f"Error getting DOM data: {e}")
            return {}
    
    def _count_buttons_with_text(self, page, text_patterns: List[str]) -> int:
        """Count buttons containing any of the given text patterns."""
        try:
            total = 0
            buttons = page.locator("button").all()
            for button in buttons:
                try:
                    btn_text = button.inner_text().lower()
                    if any(pattern.lower() in btn_text for pattern in text_patterns):
                        total += 1
                except:
                    continue
            return total
        except:
            return 0
    
    def analyze_page_state(self, page, expected_state: str = "quiz", max_text_length: int = 3000) -> PageState:
        """
        Analyze page state using DOM + Text with AI.
        
        Args:
            page: Playwright page object
            expected_state: Expected state (quiz, results, loading, etc.)
            max_text_length: Maximum text to send to AI
            
        Returns:
            PageState object with analysis results
        """
        # Layer 1: Get DOM data
        dom_data = self.get_dom_data(page)
        
        # Quick fail-fast checks before AI
        if self._quick_state_check(dom_data, expected_state):
            logger.info(f"✓ Quick check: Page appears ready for {expected_state}")
        
        # Layer 2: Get visible text
        try:
            visible_text = page.inner_text("body")[:max_text_length]
        except Exception as e:
            logger.error(f"Error getting page text: {e}")
            visible_text = "[Error getting page text]"
        
        # Layer 3: AI Analysis
        return self._ai_analyze(dom_data, visible_text, expected_state)
    
    def _quick_state_check(self, dom_data: Dict, expected_state: str) -> bool:
        """Fast heuristic check before calling AI."""
        if expected_state == "quiz":
            # Quiz should have inputs, no loading indicators
            has_inputs = (dom_data.get('radio_count', 0) + dom_data.get('checkbox_count', 0)) > 0
            no_loading = dom_data.get('loading_indicators', 0) == 0
            no_modals = dom_data.get('modal_overlays', 0) == 0
            return has_inputs and no_loading and no_modals
        
        elif expected_state == "results":
            # Results should have score pattern and retest button
            has_score = dom_data.get('has_score_pattern', False)
            has_retest = dom_data.get('retest_buttons', 0) > 0
            return has_score or has_retest
        
        return False
    
    def _ai_analyze(self, dom_data: Dict, visible_text: str, expected_state: str) -> PageState:
        """Use AI to analyze page state."""
        
        prompt = f"""You are analyzing a web page state to determine if it's ready for automation.

EXPECTED STATE: {expected_state}

DOM STRUCTURE:
- Radio buttons: {dom_data.get('radio_count', 0)}
- Checkboxes: {dom_data.get('checkbox_count', 0)}
- Labels: {dom_data.get('label_count', 0)}
- Total buttons: {dom_data.get('button_count', 0)}
- Loading indicators: {dom_data.get('loading_indicators', 0)}
- Modal overlays: {dom_data.get('modal_overlays', 0)}
- Submit buttons: {dom_data.get('submit_buttons', 0)}
- Retest buttons: {dom_data.get('retest_buttons', 0)}
- Disabled inputs: {dom_data.get('input_disabled_count', 0)}
- Has score pattern (X/Y): {dom_data.get('has_score_pattern', False)}
- Has loading text: {dom_data.get('has_loading_text', False)}
- Has error text: {dom_data.get('has_error_text', False)}

VISIBLE TEXT (first {len(visible_text)} chars):
{visible_text}

ANALYZE AND RESPOND WITH JSON ONLY (no other text):
{{
  "actual_state": "quiz|results|loading|error|login|unknown",
  "ready_to_proceed": true|false,
  "confidence": 0-100,
  "reasoning": "brief explanation of why",
  "blocking_issues": ["list any issues preventing interaction"],
  "recommended_wait_seconds": 0-5
}}

DECISION RULES:
1. QUIZ STATE requires:
   - Radio/checkboxes > 0
   - Loading indicators = 0
   - Modal overlays = 0
   - No "loading" or "please wait" text
   
2. RESULTS STATE requires:
   - Score pattern (X/Y) visible in text OR
   - Text contains "score", "result", "correct" AND
   - Retest button present OR text mentions retrying
   
3. LOADING STATE indicators:
   - Loading indicators > 0 OR
   - Text contains "loading", "please wait", "processing"
   
4. ERROR STATE indicators:
   - Text contains "error", "failed", "something went wrong"
   
5. Set ready_to_proceed = true ONLY if:
   - actual_state matches expected_state AND
   - No blocking issues AND
   - Confidence > 80

Respond with ONLY the JSON object, no markdown, no explanations."""

        response = self.ask_ai(prompt, temperature=0.1)
        
        # Parse AI response
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                
                return PageState(
                    actual_state=result.get('actual_state', 'unknown'),
                    ready_to_proceed=result.get('ready_to_proceed', False),
                    confidence=int(result.get('confidence', 0)),
                    reasoning=result.get('reasoning', 'No reasoning provided'),
                    blocking_issues=result.get('blocking_issues', []),
                    recommended_wait_seconds=int(result.get('recommended_wait_seconds', 2))
                )
            else:
                logger.error(f"No JSON found in AI response: {response[:200]}")
                return self._fallback_analysis(dom_data, expected_state)
                
        except Exception as e:
            logger.error(f"Error parsing AI response: {e}")
            logger.error(f"Response was: {response[:200]}")
            return self._fallback_analysis(dom_data, expected_state)
    
    def _fallback_analysis(self, dom_data: Dict, expected_state: str) -> PageState:
        """Fallback analysis when AI fails - use simple heuristics."""
        logger.warning("AI analysis failed, using fallback heuristics")
        
        if expected_state == "quiz":
            has_inputs = (dom_data.get('radio_count', 0) + dom_data.get('checkbox_count', 0)) > 0
            has_loading = dom_data.get('loading_indicators', 0) > 0
            has_modals = dom_data.get('modal_overlays', 0) > 0
            
            if has_inputs and not has_loading and not has_modals:
                return PageState(
                    actual_state="quiz",
                    ready_to_proceed=True,
                    confidence=70,
                    reasoning="Fallback: Has inputs, no loading/modals",
                    blocking_issues=[],
                    recommended_wait_seconds=0
                )
            else:
                issues = []
                if not has_inputs:
                    issues.append("No input elements found")
                if has_loading:
                    issues.append("Loading indicators present")
                if has_modals:
                    issues.append("Modal overlays present")
                    
                return PageState(
                    actual_state="unknown",
                    ready_to_proceed=False,
                    confidence=50,
                    reasoning="Fallback: Page not ready",
                    blocking_issues=issues,
                    recommended_wait_seconds=3
                )
        
        elif expected_state == "results":
            has_score = dom_data.get('has_score_pattern', False)
            has_retest = dom_data.get('retest_buttons', 0) > 0
            
            if has_score or has_retest:
                return PageState(
                    actual_state="results",
                    ready_to_proceed=True,
                    confidence=70,
                    reasoning="Fallback: Has score pattern or retest button",
                    blocking_issues=[],
                    recommended_wait_seconds=0
                )
        
        # Default fallback
        return PageState(
            actual_state="unknown",
            ready_to_proceed=False,
            confidence=30,
            reasoning="Fallback: Could not determine state",
            blocking_issues=["Unknown page state"],
            recommended_wait_seconds=3
        )
    
    def wait_for_state(self, page, expected_state: str, max_attempts: int = 5, 
                       min_confidence: int = 80) -> Tuple[bool, PageState]:
        """
        Wait for page to reach expected state with adaptive timing.
        
        Args:
            page: Playwright page object
            expected_state: Expected state (quiz, results, etc.)
            max_attempts: Maximum number of attempts
            min_confidence: Minimum confidence required (0-100)
            
        Returns:
            (success: bool, final_state: PageState)
        """
        logger.info(f"⏳ Waiting for page state: {expected_state}")
        
        for attempt in range(1, max_attempts + 1):
            logger.info(f"  Attempt {attempt}/{max_attempts}...")
            
            # Analyze current state
            state = self.analyze_page_state(page, expected_state)
            
            logger.info(f"  State: {state.actual_state}, Ready: {state.ready_to_proceed}, "
                       f"Confidence: {state.confidence}%")
            logger.info(f"  Reasoning: {state.reasoning}")
            
            if state.blocking_issues:
                logger.warning(f"  Blocking issues: {', '.join(state.blocking_issues)}")
            
            # Check if ready
            if state.ready_to_proceed and state.confidence >= min_confidence:
                logger.info(f"✅ Page ready! State: {state.actual_state}, Confidence: {state.confidence}%")
                return True, state
            
            # Not ready yet
            if attempt < max_attempts:
                wait_time = state.recommended_wait_seconds
                logger.info(f"  ⏱️  Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            
        # Max attempts reached
        logger.error(f"❌ Failed to reach {expected_state} state after {max_attempts} attempts")
        logger.error(f"   Final state: {state.actual_state}, Confidence: {state.confidence}%")
        return False, state
    
    def verify_quiz_ready(self, page, max_attempts: int = 5) -> bool:
        """
        Verify quiz page is ready for interaction.
        Most important verification point after retest.
        """
        logger.info("🔍 Verifying quiz page is ready...")
        success, state = self.wait_for_state(page, expected_state="quiz", 
                                             max_attempts=max_attempts, 
                                             min_confidence=80)
        
        if success:
            logger.info("✅ Quiz page verified and ready")
            return True
        else:
            logger.error("❌ Quiz page not ready")
            logger.error(f"   Blocking issues: {', '.join(state.blocking_issues)}")
            return False
    
    def verify_results_ready(self, page, max_attempts: int = 3) -> bool:
        """
        Verify results page is ready and score is visible.
        """
        logger.info("🔍 Verifying results page is ready...")
        success, state = self.wait_for_state(page, expected_state="results", 
                                             max_attempts=max_attempts, 
                                             min_confidence=75)
        
        if success:
            logger.info("✅ Results page verified and ready")
            return True
        else:
            logger.warning("⚠️  Results page verification uncertain")
            return False
    
    def get_page_screenshot_analysis(self, page) -> str:
        """
        Fallback method: Take screenshot and ask AI to analyze it.
        Only use when DOM+Text analysis fails.
        """
        try:
            import base64
            
            # Take screenshot
            screenshot_bytes = page.screenshot()
            screenshot_base64 = base64.b64encode(screenshot_bytes).decode()
            
            # Note: Ollama doesn't support vision yet, so this is a placeholder
            # for when vision models become available
            logger.warning("Screenshot analysis not yet supported by Ollama")
            return "Screenshot analysis not available"
            
        except Exception as e:
            logger.error(f"Screenshot analysis failed: {e}")
            return "Screenshot analysis failed"