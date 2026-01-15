"""AI-Powered Exam Cracker using Ollama llama3.1:8b"""

import logging
from playwright.sync_api import sync_playwright
from auth import login, navigate_to_profile
from ai_exam_solver import run_ai_solver
from config import EXAM_EMAIL, EXAM_PASSWORD, HEADLESS_MODE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point."""
    
    logger.info("="*80)
    logger.info("🤖 AI-POWERED EXAM SOLVER")
    logger.info("Using Ollama llama3.1:8b for intelligent question answering")
    logger.info("="*80)
    
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=HEADLESS_MODE)
        context = browser.new_context()
        page = context.new_page()
        
        try:
            # Step 1: Login
            logger.info("\n" + "="*80)
            logger.info("STEP 1: Authentication")
            logger.info("="*80)
            
            if not login(page, EXAM_EMAIL, EXAM_PASSWORD):
                logger.warning("⚠️ Auto-login failed")
                input("Please login manually in the browser, then press Enter...")
            else:
                logger.info("✅ Login successful")
            
            # Step 2: Navigate to profile (optional)
            logger.info("\n" + "="*80)
            logger.info("STEP 2: Profile Navigation (Optional)")
            logger.info("="*80)
            
            try:
                navigate_to_profile(page)
            except:
                logger.info("Skipping profile navigation...")
            
            # Step 3: Navigate to exam
            logger.info("\n" + "="*80)
            logger.info("STEP 3: Navigate to Exam")
            logger.info("="*80)
            logger.info(f"Current URL: {page.url}")
            logger.info("\nPlease navigate to the exam page in the browser.")
            input("Press Enter when you're on the exam page and ready to start...")
            
            exam_url = page.url
            logger.info(f"✅ Exam page: {exam_url}")
            
            # Step 4: Run AI solver
            logger.info("\n" + "="*80)
            logger.info("STEP 4: AI Exam Solving")
            logger.info("="*80)
            
            # Ask user for number of questions
            num_q = input("How many questions in this exam? (default: 10): ").strip()
            num_questions = int(num_q) if num_q.isdigit() else 10
            
            logger.info(f"Solving exam with {num_questions} questions...")
            run_ai_solver(page, num_questions=num_questions, max_attempts=20)
            
            # Keep browser open for review
            logger.info("\n" + "="*80)
            logger.info("✅ Exam solving complete!")
            logger.info("Browser will remain open for review.")
            logger.info("="*80)
            input("\nPress Enter to close browser and exit...")
            
        except KeyboardInterrupt:
            logger.info("\n⚠️ Interrupted by user")
        except Exception as e:
            logger.error(f"❌ Error: {e}", exc_info=True)
            input("Press Enter to exit...")
        finally:
            try:
                browser.close()
                logger.info("Browser closed")
            except:
                pass


if __name__ == "__main__":
    main()