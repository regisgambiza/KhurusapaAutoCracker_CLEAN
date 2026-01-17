"""
Beautiful Tkinter GUI for AI-Powered Exam Solver
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import logging
from datetime import datetime
from playwright.sync_api import sync_playwright
from auth import login, navigate_to_profile
from ai_exam_solver import run_ai_solver
from config import EXAM_EMAIL, EXAM_PASSWORD, HEADLESS_MODE
import sys
from io import StringIO


class TextHandler(logging.Handler):
    """Custom logging handler to redirect logs to GUI"""
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget
        
    def emit(self, record):
        msg = self.format(record)
        def append():
            self.text_widget.configure(state='normal')
            self.text_widget.insert(tk.END, msg + '\n')
            self.text_widget.configure(state='disabled')
            self.text_widget.see(tk.END)
        self.text_widget.after(0, append)


class ExamSolverGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🤖 AI-Powered Exam Solver")
        self.root.geometry("1000x700")
        self.root.configure(bg="#1e1e1e")
        
        # Configure style
        self.setup_styles()
        
        # State variables
        self.is_running = False
        self.solver_thread = None
        
        # Build UI
        self.build_ui()
        
        # Setup logging
        self.setup_logging()
        
    def setup_styles(self):
        """Configure custom styles"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        bg_dark = "#1e1e1e"
        bg_medium = "#2d2d2d"
        bg_light = "#3d3d3d"
        accent = "#007acc"
        accent_hover = "#005a9e"
        text_color = "#ffffff"
        
        # Frame style
        style.configure("Dark.TFrame", background=bg_dark)
        style.configure("Medium.TFrame", background=bg_medium)
        
        # Label style
        style.configure("Dark.TLabel", 
                       background=bg_dark, 
                       foreground=text_color,
                       font=('Segoe UI', 10))
        style.configure("Title.TLabel",
                       background=bg_dark,
                       foreground=accent,
                       font=('Segoe UI', 16, 'bold'))
        style.configure("Header.TLabel",
                       background=bg_medium,
                       foreground=text_color,
                       font=('Segoe UI', 11, 'bold'))
        
        # Button style
        style.configure("Accent.TButton",
                       background=accent,
                       foreground=text_color,
                       borderwidth=0,
                       focuscolor='none',
                       font=('Segoe UI', 10, 'bold'),
                       padding=10)
        style.map("Accent.TButton",
                 background=[('active', accent_hover)])
        
        # Entry style
        style.configure("Dark.TEntry",
                       fieldbackground=bg_light,
                       foreground=text_color,
                       borderwidth=1,
                       insertcolor=text_color)
        
    def build_ui(self):
        """Build the GUI interface"""
        # Main container
        main_frame = ttk.Frame(self.root, style="Dark.TFrame", padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_frame = ttk.Frame(main_frame, style="Dark.TFrame")
        title_frame.pack(fill=tk.X, pady=(0, 20))
        
        title_label = ttk.Label(title_frame, 
                               text="🤖 AI-Powered Exam Solver",
                               style="Title.TLabel")
        title_label.pack()
        
        subtitle = ttk.Label(title_frame,
                            text="Using Ollama llama3.1:8b for intelligent solving",
                            style="Dark.TLabel")
        subtitle.pack()
        
        # Configuration section
        config_frame = ttk.LabelFrame(main_frame, text="Configuration", 
                                     style="Medium.TFrame", padding=15)
        config_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Email
        email_frame = ttk.Frame(config_frame, style="Medium.TFrame")
        email_frame.pack(fill=tk.X, pady=5)
        ttk.Label(email_frame, text="Email:", style="Dark.TLabel", width=15).pack(side=tk.LEFT)
        self.email_var = tk.StringVar(value=EXAM_EMAIL)
        email_entry = ttk.Entry(email_frame, textvariable=self.email_var, 
                               style="Dark.TEntry", width=40)
        email_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Password
        pass_frame = ttk.Frame(config_frame, style="Medium.TFrame")
        pass_frame.pack(fill=tk.X, pady=5)
        ttk.Label(pass_frame, text="Password:", style="Dark.TLabel", width=15).pack(side=tk.LEFT)
        self.password_var = tk.StringVar(value=EXAM_PASSWORD)
        pass_entry = ttk.Entry(pass_frame, textvariable=self.password_var, 
                              show="*", style="Dark.TEntry", width=40)
        pass_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Number of questions
        num_q_frame = ttk.Frame(config_frame, style="Medium.TFrame")
        num_q_frame.pack(fill=tk.X, pady=5)
        ttk.Label(num_q_frame, text="Questions:", style="Dark.TLabel", width=15).pack(side=tk.LEFT)
        self.num_questions_var = tk.StringVar(value="10")
        num_q_entry = ttk.Entry(num_q_frame, textvariable=self.num_questions_var,
                               style="Dark.TEntry", width=10)
        num_q_entry.pack(side=tk.LEFT, padx=5)
        
        # Max attempts
        attempts_frame = ttk.Frame(config_frame, style="Medium.TFrame")
        attempts_frame.pack(fill=tk.X, pady=5)
        ttk.Label(attempts_frame, text="Max Attempts:", style="Dark.TLabel", width=15).pack(side=tk.LEFT)
        self.max_attempts_var = tk.StringVar(value="100")
        attempts_entry = ttk.Entry(attempts_frame, textvariable=self.max_attempts_var,
                                  style="Dark.TEntry", width=10)
        attempts_entry.pack(side=tk.LEFT, padx=5)
        
        # Status section
        status_frame = ttk.LabelFrame(main_frame, text="Status", 
                                     style="Medium.TFrame", padding=15)
        status_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.status_label = ttk.Label(status_frame, 
                                      text="Ready to start",
                                      style="Header.TLabel")
        self.status_label.pack()
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(status_frame, 
                                           variable=self.progress_var,
                                           mode='indeterminate',
                                           length=300)
        self.progress_bar.pack(pady=10)
        
        # Control buttons
        button_frame = ttk.Frame(main_frame, style="Dark.TFrame")
        button_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.start_button = ttk.Button(button_frame, 
                                       text="🚀 Start Solving",
                                       style="Accent.TButton",
                                       command=self.start_solving)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(button_frame,
                                      text="⏹ Stop",
                                      style="Accent.TButton",
                                      command=self.stop_solving,
                                      state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        self.continue_button = ttk.Button(button_frame,
                                         text="▶️ Continue",
                                         style="Accent.TButton",
                                         command=self.continue_process,
                                         state=tk.DISABLED)
        self.continue_button.pack(side=tk.LEFT, padx=5)
        
        clear_button = ttk.Button(button_frame,
                                 text="🗑 Clear Log",
                                 style="Accent.TButton",
                                 command=self.clear_log)
        clear_button.pack(side=tk.LEFT, padx=5)
        
        # Log section
        log_frame = ttk.LabelFrame(main_frame, text="Log Output",
                                  style="Medium.TFrame", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        self.log_text = scrolledtext.ScrolledText(log_frame,
                                                  wrap=tk.WORD,
                                                  bg="#0c0c0c",
                                                  fg="#00ff00",
                                                  font=('Consolas', 9),
                                                  insertbackground="#00ff00",
                                                  state='disabled')
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
    def setup_logging(self):
        """Setup logging to redirect to GUI"""
        # Create logger
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Add GUI handler
        gui_handler = TextHandler(self.log_text)
        gui_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                            datefmt='%H:%M:%S')
        )
        logger.addHandler(gui_handler)
        
    def start_solving(self):
        """Start the exam solving process"""
        if self.is_running:
            messagebox.showwarning("Warning", "Solver is already running!")
            return
        
        # Validate inputs
        try:
            num_questions = int(self.num_questions_var.get())
            max_attempts = int(self.max_attempts_var.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers!")
            return
        
        if not self.email_var.get() or not self.password_var.get():
            messagebox.showerror("Error", "Please enter email and password!")
            return
        
        # Update UI
        self.is_running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_label.config(text="🔄 Solving in progress...")
        self.progress_bar.start(10)
        
        # Start solver in separate thread
        self.solver_thread = threading.Thread(
            target=self.run_solver,
            args=(num_questions, max_attempts),
            daemon=True
        )
        self.solver_thread.start()
        
    def stop_solving(self):
        """Stop the solving process"""
        if messagebox.askyesno("Confirm", "Are you sure you want to stop?"):
            self.is_running = False
            self.reset_ui()
            logging.info("⚠️ Stopped by user")
    
    def continue_process(self):
        """Continue button handler - signals waiting thread to proceed"""
        self._user_confirmed = True
        self.continue_button.config(state=tk.DISABLED)
        self.status_label.config(text="🔄 Processing...")
        logging.info("▶️ Continue button clicked, proceeding...")
        
    def reset_ui(self):
        """Reset UI to initial state"""
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.continue_button.config(state=tk.DISABLED)
        self.status_label.config(text="Ready to start")
        self.progress_bar.stop()
        
    def clear_log(self):
        """Clear the log text"""
        self.log_text.configure(state='normal')
        self.log_text.delete(1.0, tk.END)
        self.log_text.configure(state='disabled')
        
    def run_solver(self, num_questions, max_attempts):
        """Run the exam solver (in separate thread)"""
        try:
            logging.info("="*80)
            logging.info("🤖 AI-POWERED EXAM SOLVER STARTED")
            logging.info("="*80)
            
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=HEADLESS_MODE)
                context = browser.new_context()
                page = context.new_page()
                
                try:
                    # Login
                    logging.info("STEP 1: Authentication")
                    if login(page, self.email_var.get(), self.password_var.get()):
                        logging.info("✅ Login successful")
                    else:
                        logging.warning("⚠️ Auto-login failed")
                    
                    # Navigate to profile
                    try:
                        navigate_to_profile(page)
                    except:
                        logging.info("Skipping profile navigation...")
                    
                    # Wait for manual navigation with continue button
                    logging.info("="*80)
                    logging.info("STEP 2: Manual Navigation Required")
                    logging.info("="*80)
                    logging.info("📍 Please navigate to the EXAM/QUIZ page in the browser")
                    logging.info("The page should contain the actual quiz questions")
                    logging.info("Once you're on the exam page, click the CONTINUE button")
                    
                    # Enable continue button and wait
                    self.root.after(0, self._enable_continue_button)
                    self._user_confirmed = False
                    
                    # Wait for user to click Continue
                    import time
                    while not self._user_confirmed and self.is_running:
                        time.sleep(0.2)
                    
                    if not self.is_running:
                        logging.info("⚠️ Process stopped by user")
                        return
                    
                    logging.info(f"Current URL: {page.url}")
                    logging.info("Starting question extraction...")
                    
                    # Verify we're on the right page
                    try:
                        # Check if quiz elements exist
                        page.wait_for_selector("input[type='radio'], input[type='checkbox']", timeout=5000)
                        logging.info("✅ Quiz elements detected!")
                    except:
                        logging.error("❌ No quiz elements found on this page!")
                        logging.error("Please make sure you're on the actual exam/quiz page")
                        logging.error("The page should show questions with radio buttons or checkboxes")
                        
                        self.root.after(0, lambda: messagebox.showerror(
                            "Wrong Page",
                            "Could not find quiz elements on this page!\n\n"
                            "Please navigate to the actual EXAM/QUIZ page where you can see:\n"
                            "  • Questions\n"
                            "  • Answer options (A, B, C, D)\n"
                            "  • Radio buttons or checkboxes\n\n"
                            "Then restart the solver."
                        ))
                        return
                    
                    # Run solver
                    logging.info("STEP 2: Starting AI Solver")
                    run_ai_solver(page, num_questions=num_questions, max_attempts=max_attempts)
                    
                    logging.info("="*80)
                    logging.info("✅ EXAM SOLVING COMPLETED")
                    logging.info("="*80)
                    
                    self.root.after(0, lambda: self.on_completion(True))
                    
                except Exception as e:
                    logging.error(f"❌ Error: {e}", exc_info=True)
                    self.root.after(0, lambda: self.on_completion(False))
                finally:
                    # Keep browser open
                    logging.info("Browser remains open for review")
                    logging.info("Close the browser window when done")
                    
        except Exception as e:
            logging.error(f"❌ Fatal error: {e}", exc_info=True)
            self.root.after(0, lambda: self.on_completion(False))
        finally:
            self.is_running = False
            self.root.after(0, self.reset_ui)
    
    def on_completion(self, success):
        """Handle solver completion"""
        if success:
            self.status_label.config(text="✅ Completed successfully!")
            messagebox.showinfo("Success", "Exam solving completed!\n\nCheck the log for details.")
        else:
            self.status_label.config(text="❌ Completed with errors")
            messagebox.showwarning("Warning", "Solver completed with errors.\n\nCheck the log for details.")
    
    def _show_navigation_dialog(self):
        """Show navigation dialog and set confirmation flag"""
        messagebox.showinfo(
            "📍 Navigate to Exam Page",
            "Please navigate to the EXAM/QUIZ page in the browser window.\n\n"
            "Make sure you can see the actual quiz questions with:\n"
            "  ✓ Radio buttons or checkboxes\n"
            "  ✓ Question text\n"
            "  ✓ Answer options\n\n"
            "Click OK when you're on the exam page and ready to start.",
            icon='info'
        )
        self._navigation_confirmed = True
    
    def _enable_continue_button(self):
        """Enable the continue button and update status"""
        self.continue_button.config(state=tk.NORMAL)
        self.status_label.config(text="⏸️ Waiting for user - Click CONTINUE")
        
        # Make continue button more visible
        self.continue_button.config(
            style="Accent.TButton"
        )
    
    def _wait_for_user(self, message: str):
        """Generic method to wait for user confirmation via Continue button"""
        logging.info("="*60)
        logging.info(f"⏸️ {message}")
        logging.info("Click the CONTINUE button when ready")
        logging.info("="*60)
        
        self._user_confirmed = False
        self.root.after(0, self._enable_continue_button)
        
        import time
        while not self._user_confirmed and self.is_running:
            time.sleep(0.2)
        
        return self.is_running  # Return False if stopped


def main():
    """Main entry point for GUI"""
    root = tk.Tk()
    app = ExamSolverGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()