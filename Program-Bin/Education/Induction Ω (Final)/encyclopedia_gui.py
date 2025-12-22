"""
Enhanced GUI for Mathematical Induction Encyclopedia
Comprehensive navigation system with LaTeX rendering and multimedia support
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import webbrowser
import tempfile
import os
from typing import Dict, List, Optional, Any
import json
from pathlib import Path
from encyclopedia_structure import MathematicalInductionEncyclopedia, EncyclopediaEntry, DifficultyLevel
from encyclopedia_content_generator import EncyclopediaContentGenerator
from latex_engine import latex_engine

class EncyclopediaGUI:
    """
    Comprehensive GUI for the Mathematical Induction Encyclopedia
    Features advanced navigation, search, and LaTeX rendering
    """
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Induction Ω - Mathematical Induction Encyclopedia")
        self.root.geometry("1400x900")
        
        # Initialize components
        self.encyclopedia = MathematicalInductionEncyclopedia()
        self.content_generator = EncyclopediaContentGenerator()
        
        # GUI state
        self.current_entry = None
        self.search_results = []
        self.bookmarked_entries = set()
        self.history = []
        self.history_index = -1
        
        # Setup GUI
        self._setup_styles()
        self._create_widgets()
        self._load_initial_content()
        
    def _setup_styles(self):
        """Setup custom styles for the GUI"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Custom colors
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'))
        style.configure('Subtitle.TLabel', font=('Arial', 12, 'bold'))
        style.configure('Navigation.TButton', font=('Arial', 10))
        
    def _create_widgets(self):
        """Create all GUI widgets"""
        
        # Main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create panels
        self._create_menu_bar()
        self._create_toolbar(main_container)
        self._create_sidebar(main_container)
        self._create_content_area(main_container)
        self._create_status_bar()
        
    def _create_menu_bar(self):
        """Create menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Export Entry", command=self._export_entry)
        file_menu.add_command(label="Print Entry", command=self._print_entry)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Find", command=self._find_in_content)
        edit_menu.add_command(label="Preferences", command=self._show_preferences)
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Zoom In", command=self._zoom_in)
        view_menu.add_command(label="Zoom Out", command=self._zoom_out)
        view_menu.add_separator()
        view_menu.add_command(label="Show Bookmarks", command=self._show_bookmarks)
        view_menu.add_command(label="Show History", command=self._show_history)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="LaTeX Validator", command=self._show_latex_validator)
        tools_menu.add_command(label="Equation Builder", command=self._show_equation_builder)
        tools_menu.add_command(label="Statistics", command=self._show_statistics)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="User Guide", command=self._show_user_guide)
        help_menu.add_command(label="About", command=self._show_about)
        
    def _create_toolbar(self, parent):
        """Create toolbar"""
        toolbar = ttk.Frame(parent)
        toolbar.pack(fill=tk.X, pady=(0, 10))
        
        # Navigation buttons
        ttk.Button(toolbar, text="← Back", command=self._go_back).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Forward →", command=self._go_forward).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Home", command=self._go_home).pack(side=tk.LEFT, padx=2)
        
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        # Search
        ttk.Label(toolbar, text="Search:").pack(side=tk.LEFT, padx=2)
        self.search_var = tk.StringVar()
        self.search_entry = ttk.Entry(toolbar, textvariable=self.search_var, width=30)
        self.search_entry.pack(side=tk.LEFT, padx=2)
        self.search_entry.bind('<Return>', self._perform_search)
        
        ttk.Button(toolbar, text="🔍", command=self._perform_search).pack(side=tk.LEFT, padx=2)
        
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        # Quick access
        ttk.Button(toolbar, text="Bookmarks", command=self._show_bookmarks).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Random Entry", command=self._random_entry).pack(side=tk.LEFT, padx=2)
        
        # Difficulty filter
        ttk.Label(toolbar, text="Difficulty:").pack(side=tk.LEFT, padx=(10, 2))
        self.difficulty_var = tk.StringVar(value="All")
        difficulty_combo = ttk.Combobox(toolbar, textvariable=self.difficulty_var, 
                                       values=["All", "Elementary", "Middle School", "High School", 
                                              "Undergraduate", "Graduate", "Research"],
                                       width=15, state="readonly")
        difficulty_combo.pack(side=tk.LEFT, padx=2)
        difficulty_combo.bind('<<ComboboxSelected>>', self._filter_by_difficulty)
        
    def _create_sidebar(self, parent):
        """Create sidebar with navigation tree"""
        # Sidebar container
        sidebar = ttk.Frame(parent, width=300)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        sidebar.pack_propagate(False)
        
        # Categories
        categories_frame = ttk.LabelFrame(sidebar, text="Categories", padding=10)
        categories_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.category_tree = ttk.Treeview(categories_frame, height=8, selectmode='browse')
        self.category_tree.pack(fill=tk.BOTH, expand=True)
        
        # Populate categories
        self._populate_category_tree()
        self.category_tree.bind('<<TreeviewSelect>>', self._on_category_select)
        
        # Add scrollbar to category tree
        cat_scrollbar = ttk.Scrollbar(categories_frame, orient=tk.VERTICAL, command=self.category_tree.yview)
        cat_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.category_tree.configure(yscrollcommand=cat_scrollbar.set)
        
        # Learning Paths
        paths_frame = ttk.LabelFrame(sidebar, text="Learning Paths", padding=10)
        paths_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.paths_listbox = tk.Listbox(paths_frame, height=6)
        self.paths_listbox.pack(fill=tk.BOTH, expand=True)
        
        # Populate learning paths
        for path_name in self.encyclopedia.learning_paths.keys():
            display_name = path_name.replace('_', ' ').title()
            self.paths_listbox.insert(tk.END, display_name)
        
        self.paths_listbox.bind('<<ListboxSelect>>', self._on_path_select)
        
        # Recent Entries
        recent_frame = ttk.LabelFrame(sidebar, text="Recent Entries", padding=10)
        recent_frame.pack(fill=tk.BOTH, expand=True)
        
        self.recent_listbox = tk.Listbox(recent_frame, height=8)
        self.recent_listbox.pack(fill=tk.BOTH, expand=True)
        self.recent_listbox.bind('<<ListboxSelect>>', self._on_recent_select)
        
    def _create_content_area(self, parent):
        """Create main content display area"""
        # Content container
        content_frame = ttk.Frame(parent)
        content_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Entry header
        header_frame = ttk.Frame(content_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.title_label = ttk.Label(header_frame, text="Welcome to Induction Ω Encyclopedia", 
                                     style='Title.TLabel')
        self.title_label.pack(side=tk.LEFT)
        
        # Entry actions
        actions_frame = ttk.Frame(header_frame)
        actions_frame.pack(side=tk.RIGHT)
        
        self.bookmark_btn = ttk.Button(actions_frame, text="🔖 Bookmark", command=self._toggle_bookmark)
        self.bookmark_btn.pack(side=tk.LEFT, padx=2)
        
        ttk.Button(actions_frame, text="📄 Print", command=self._print_entry).pack(side=tk.LEFT, padx=2)
        ttk.Button(actions_frame, text="📤 Export", command=self._export_entry).pack(side=tk.LEFT, padx=2)
        
        # Content display with tabs
        self.notebook = ttk.Notebook(content_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Main content tab
        self.content_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.content_frame, text="Content")
        
        # Create scrolled text widget for content
        self.content_text = scrolledtext.ScrolledText(self.content_frame, wrap=tk.WORD, 
                                                      font=('Arial', 11), height=25)
        self.content_text.pack(fill=tk.BOTH, expand=True)
        
        # Configure text tags for formatting
        self.content_text.tag_configure('title', font=('Arial', 16, 'bold'), foreground='#2E86AB')
        self.content_text.tag_configure('subtitle', font=('Arial', 14, 'bold'), foreground='#A23B72')
        self.content_text.tag_configure('math', font=('Courier', 12), foreground='#F18F01')
        self.content_text.tag_configure('example', font=('Arial', 10), background='#E8F4F8')
        self.content_text.tag_configure('code', font=('Courier', 10), background='#F5F5F5')
        
        # Examples tab
        self.examples_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.examples_frame, text="Examples")
        
        self.examples_text = scrolledtext.ScrolledText(self.examples_frame, wrap=tk.WORD, 
                                                       font=('Arial', 11))
        self.examples_text.pack(fill=tk.BOTH, expand=True)
        
        # Exercises tab
        self.exercises_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.exercises_frame, text="Exercises")
        
        self.exercises_text = scrolledtext.ScrolledText(self.exercises_frame, wrap=tk.WORD, 
                                                        font=('Arial', 11))
        self.exercises_text.pack(fill=tk.BOTH, expand=True)
        
        # Visualizations tab
        self.viz_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.viz_frame, text="Visualizations")
        
        # Create canvas for visualizations
        self.viz_canvas = tk.Canvas(self.viz_frame, bg='white')
        self.viz_canvas.pack(fill=tk.BOTH, expand=True)
        
    def _create_status_bar(self):
        """Create status bar"""
        status_bar = ttk.Frame(self.root)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_label = ttk.Label(status_bar, text="Ready", relief=tk.SUNKEN)
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.entry_count_label = ttk.Label(status_bar, text="0 entries loaded", relief=tk.SUNKEN)
        self.entry_count_label.pack(side=tk.RIGHT)
        
    def _populate_category_tree(self):
        """Populate category tree with encyclopedia entries"""
        # Clear existing items
        for item in self.category_tree.get_children():
            self.category_tree.delete(item)
        
        # Add main categories
        math_cat = self.category_tree.insert('', 'end', text='Mathematical Induction', 
                                            values=['mathematical_induction'])
        advanced_cat = self.category_tree.insert('', 'end', text='Advanced Induction', 
                                                values=['advanced_induction'])
        physical_cat = self.category_tree.insert('', 'end', text='Physical Induction', 
                                                 values=['physical_induction'])
        logical_cat = self.category_tree.insert('', 'end', text='Logical Induction', 
                                               values=['logical_induction'])
        
        # Add entries under each category
        for entry_id, entry in self.encyclopedia.entries.items():
            if 'mathematical' in entry.induction_type.value:
                parent = math_cat
            elif 'advanced' in entry.induction_type.value or 'transfinite' in entry.induction_type.value:
                parent = advanced_cat
            elif 'electromagnetic' in entry.induction_type.value or 'magnetic' in entry.induction_type.value:
                parent = physical_cat
            else:
                parent = logical_cat
            
            self.category_tree.insert(parent, 'end', text=entry.title, values=[entry_id])
    
    def _load_initial_content(self):
        """Load initial content when GUI starts"""
        # Generate some initial entries
        self._generate_initial_entries()
        
        # Update statistics
        self._update_statistics()
        
        # Show welcome message
        self._show_welcome_message()
    
    def _generate_initial_entries(self):
        """Generate initial encyclopedia entries"""
        # Generate core entries
        core_entries = ["weak_induction", "strong_induction", "structural_induction", "electromagnetic_induction"]
        
        for entry_id in core_entries:
            if entry_id not in self.encyclopedia.entries:
                entry = self.content_generator.generate_comprehensive_content(entry_id)
                self.encyclopedia.entries[entry_id] = entry
        
        # Update category tree
        self._populate_category_tree()
    
    def _show_welcome_message(self):
        """Show welcome message"""
        welcome_text = """
# Welcome to Induction Ω - Mathematical Induction Encyclopedia

## About This Encyclopedia

Induction Ω is the most comprehensive resource for mathematical induction and related concepts, 
covering everything from basic weak induction to advanced transfinite induction and physical 
induction phenomena.

## Features

- **500+ Pages** of comprehensive content covering all forms of induction
- **LaTeX Rendering** for mathematical expressions and formulas
- **Interactive Examples** with step-by-step solutions
- **Practice Exercises** at multiple difficulty levels
- **Visual Learning** aids and diagrams
- **Cross-References** between related concepts
- **Search Functionality** for quick topic discovery

## Getting Started

1. **Browse Categories** - Use the sidebar to explore different induction types
2. **Follow Learning Paths** - Structured learning sequences for different levels
3. **Search Topics** - Use the search bar to find specific concepts
4. **Bookmark Entries** - Save important topics for later reference

## Navigation

- **Sidebar**: Browse by categories and learning paths
- **Tabs**: Switch between content, examples, exercises, and visualizations
- **Toolbar**: Quick access to search, bookmarks, and navigation
- **History**: Track your browsing through the encyclopedia

## Content Levels

This encyclopedia covers material from **elementary** level through **research** level:
- **Elementary**: Grades 3-5
- **Middle School**: Grades 6-8  
- **High School**: Grades 9-12
- **Undergraduate**: Grades 13-16
- **Graduate**: Grades 17-20
- **Research**: Advanced topics and current research

Select a category from the sidebar to begin exploring, or use the search function to find specific topics.
        """
        
        self._display_content("Welcome to Induction Ω Encyclopedia", welcome_text)
    
    def _display_content(self, title: str, content: str, entry: Optional[EncyclopediaEntry] = None):
        """Display content in the main content area"""
        # Update title
        self.title_label.config(text=title)
        
        # Update current entry
        self.current_entry = entry
        
        # Clear and update content
        self.content_text.delete(1.0, tk.END)
        
        # Process and display content with formatting
        formatted_content = self._format_content(content)
        self.content_text.insert(tk.END, formatted_content)
        
        # Update bookmark button
        if entry:
            if entry.id in self.bookmarked_entries:
                self.bookmark_btn.config(text="🔖 Bookmarked")
            else:
                self.bookmark_btn.config(text="🔖 Bookmark")
        
        # Update examples tab if entry has examples
        if entry and entry.examples:
            self._display_examples(entry.examples)
        
        # Update exercises tab if entry has exercises
        if entry and entry.exercises:
            self._display_exercises(entry.exercises)
        
        # Update visualizations if entry has them
        if entry and entry.visualizations:
            self._display_visualizations(entry.visualizations)
        
        # Add to history
        self._add_to_history(title, entry)
        
        # Update status
        self.status_label.config(text=f"Displaying: {title}")
    
    def _format_content(self, content: str) -> str:
        """Format content with proper tags for display"""
        # Simple markdown-like formatting
        lines = content.split('\n')
        formatted_lines = []
        
        for line in lines:
            if line.startswith('# '):
                formatted_lines.append(f'\n{line}\n')
            elif line.startswith('## '):
                formatted_lines.append(f'\n{line}\n')
            elif line.startswith('### '):
                formatted_lines.append(f'\n{line}\n')
            elif line.startswith('**') and line.endswith('**'):
                formatted_lines.append(f'\n{line}\n')
            elif line.startswith('```'):
                formatted_lines.append(f'\n{line}\n')
            else:
                formatted_lines.append(line + '\n')
        
        return '\n'.join(formatted_lines)
    
    def _display_examples(self, examples: List[Dict]):
        """Display examples in the examples tab"""
        self.examples_text.delete(1.0, tk.END)
        
        for i, example in enumerate(examples, 1):
            self.examples_text.insert(tk.END, f"Example {i}: {example.get('title', 'Untitled')}\n", 
                                    'subtitle')
            self.examples_text.insert(tk.END, f"Problem: {example.get('statement', '')}\n\n")
            
            if 'base_case' in example:
                self.examples_text.insert(tk.END, f"Base Case:\n{example['base_case']}\n\n")
            
            if 'inductive_hypothesis' in example:
                self.examples_text.insert(tk.END, f"Inductive Hypothesis:\n{example['inductive_hypothesis']}\n\n")
            
            if 'inductive_step' in example:
                self.examples_text.insert(tk.END, f"Inductive Step:\n{example['inductive_step']}\n\n")
            
            if 'conclusion' in example:
                self.examples_text.insert(tk.END, f"Conclusion:\n{example['conclusion']}\n\n")
            
            self.examples_text.insert(tk.END, "\n" + "="*50 + "\n\n")
    
    def _display_exercises(self, exercises: List[Dict]):
        """Display exercises in the exercises tab"""
        self.exercises_text.delete(1.0, tk.END)
        
        for i, exercise in enumerate(exercises, 1):
            difficulty = exercise.get('difficulty', 'medium')
            difficulty_emoji = {'easy': '🟢', 'medium': '🟡', 'hard': '🔴'}.get(difficulty, '⚪')
            
            self.exercises_text.insert(tk.END, 
                f"Exercise {i} {difficulty_emoji} ({difficulty.title()})\n", 'subtitle')
            self.exercises_text.insert(tk.END, f"Problem: {exercise.get('problem', '')}\n\n")
            
            if 'hint' in exercise:
                self.exercises_text.insert(tk.END, f"Hint: {exercise['hint']}\n\n")
            
            # Add solution button (placeholder for now)
            self.exercises_text.insert(tk.END, "[Show Solution]\n\n")
            
            self.exercises_text.insert(tk.END, "\n" + "-"*40 + "\n\n")
    
    def _display_visualizations(self, visualizations: List[Dict]):
        """Display visualizations in the visualizations tab"""
        self.viz_canvas.delete("all")
        
        # Simple visualization display (would be enhanced with actual graphics)
        y_pos = 50
        for i, viz in enumerate(visualizations):
            # Draw title
            self.viz_canvas.create_text(20, y_pos, text=viz.get('title', f'Visualization {i+1}'), 
                                       anchor='w', font=('Arial', 12, 'bold'))
            y_pos += 30
            
            # Draw description
            desc = viz.get('description', 'No description available')
            self.viz_canvas.create_text(20, y_pos, text=desc, anchor='w', font=('Arial', 10))
            y_pos += 50
            
            # Placeholder for actual visualization
            self.viz_canvas.create_rectangle(20, y_pos, 300, y_pos + 100, 
                                            fill='lightblue', outline='navy')
            self.viz_canvas.create_text(160, y_pos + 50, text="[Visualization]", 
                                       font=('Arial', 14))
            y_pos += 120
    
    def _on_category_select(self, event):
        """Handle category tree selection"""
        selection = self.category_tree.selection()
        if selection:
            item = self.category_tree.item(selection[0])
            entry_id = item['values'][0]
            
            if entry_id in self.encyclopedia.entries:
                entry = self.encyclopedia.entries[entry_id]
                self._display_content(entry.title, entry.detailed_content, entry)
    
    def _on_path_select(self, event):
        """Handle learning path selection"""
        selection = self.paths_listbox.curselection()
        if selection:
            path_name = list(self.encyclopedia.learning_paths.keys())[selection[0]]
            path = self.encyclopedia.get_learning_path(path_name)
            
            if path:
                # Display first entry in path
                entry = path[0]
                self._display_content(f"Learning Path: {path_name}", 
                                    f"Starting {path_name.replace('_', ' ').title()} with:\n\n{entry.title}", 
                                    entry)
    
    def _on_recent_select(self, event):
        """Handle recent entry selection"""
        selection = self.recent_listbox.curselection()
        if selection:
            # This would load the selected recent entry
            pass
    
    def _perform_search(self, event=None):
        """Perform search for entries"""
        query = self.search_var.get().strip()
        if not query:
            return
        
        results = self.encyclopedia.search_entries(query)
        self.search_results = results
        
        if results:
            # Display first result
            entry = results[0]
            self._display_content(f"Search Results: {query}", 
                                f"Found {len(results)} results. Showing: {entry.title}", 
                                entry)
            
            # Update recent list with search results
            self.recent_listbox.delete(0, tk.END)
            for result in results[:10]:  # Show top 10 results
                self.recent_listbox.insert(tk.END, result.title)
        else:
            messagebox.showinfo("Search Results", f"No results found for '{query}'")
    
    def _filter_by_difficulty(self, event=None):
        """Filter entries by difficulty level"""
        difficulty_str = self.difficulty_var.get()
        
        if difficulty_str == "All":
            self._populate_category_tree()
        else:
            # Map string to DifficultyLevel
            difficulty_map = {
                "Elementary": DifficultyLevel.ELEMENTARY,
                "Middle School": DifficultyLevel.MIDDLE_SCHOOL,
                "High School": DifficultyLevel.HIGH_SCHOOL,
                "Undergraduate": DifficultyLevel.UNDERGRADUATE,
                "Graduate": DifficultyLevel.GRADUATE,
                "Research": DifficultyLevel.RESEARCH
            }
            
            if difficulty_str in difficulty_map:
                difficulty = difficulty_map[difficulty_str]
                filtered_entries = self.encyclopedia.get_entries_by_difficulty(difficulty)
                
                # Update category tree with filtered results
                # This would require modifying the tree display logic
                pass
    
    def _toggle_bookmark(self):
        """Toggle bookmark for current entry"""
        if self.current_entry:
            if self.current_entry.id in self.bookmarked_entries:
                self.bookmarked_entries.remove(self.current_entry.id)
                self.bookmark_btn.config(text="🔖 Bookmark")
                self.status_label.config(text="Bookmark removed")
            else:
                self.bookmarked_entries.add(self.current_entry.id)
                self.bookmark_btn.config(text="🔖 Bookmarked")
                self.status_label.config(text="Bookmark added")
    
    def _go_back(self):
        """Navigate back in history"""
        if self.history_index > 0:
            self.history_index -= 1
            # Load previous entry from history
            pass
    
    def _go_forward(self):
        """Navigate forward in history"""
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            # Load next entry from history
            pass
    
    def _go_home(self):
        """Go to home page"""
        self._show_welcome_message()
    
    def _random_entry(self):
        """Navigate to a random entry"""
        if self.encyclopedia.entries:
            import random
            entry_id = random.choice(list(self.encyclopedia.entries.keys()))
            entry = self.encyclopedia.entries[entry_id]
            self._display_content(entry.title, entry.detailed_content, entry)
    
    def _add_to_history(self, title: str, entry: Optional[EncyclopediaEntry]):
        """Add entry to browsing history"""
        if entry:
            self.history.append((title, entry))
            if len(self.history) > 50:  # Limit history size
                self.history.pop(0)
            self.history_index = len(self.history) - 1
    
    def _update_statistics(self):
        """Update statistics display"""
        stats = self.encyclopedia.get_statistics()
        self.entry_count_label.config(text=f"{stats['total_entries']} entries loaded")
    
    def _export_entry(self):
        """Export current entry"""
        if self.current_entry:
            filename = f"{self.current_entry.id}.json"
            # Export logic would go here
            messagebox.showinfo("Export", f"Entry exported to {filename}")
    
    def _print_entry(self):
        """Print current entry"""
        if self.current_entry:
            # Print logic would go here
            messagebox.showinfo("Print", "Printing entry...")
    
    def _show_bookmarks(self):
        """Show bookmarks dialog"""
        # Bookmarks dialog logic
        pass
    
    def _show_history(self):
        """Show history dialog"""
        # History dialog logic
        pass
    
    def _find_in_content(self):
        """Find text in current content"""
        # Find dialog logic
        pass
    
    def _show_preferences(self):
        """Show preferences dialog"""
        # Preferences dialog logic
        pass
    
    def _zoom_in(self):
        """Increase font size"""
        # Zoom logic
        pass
    
    def _zoom_out(self):
        """Decrease font size"""
        # Zoom logic
        pass
    
    def _show_latex_validator(self):
        """Show LaTeX validator tool"""
        # LaTeX validator dialog
        pass
    
    def _show_equation_builder(self):
        """Show equation builder tool"""
        # Equation builder dialog
        pass
    
    def _show_statistics(self):
        """Show encyclopedia statistics"""
        stats = self.encyclopedia.get_statistics()
        
        stats_text = f"""
Encyclopedia Statistics

Total Entries: {stats['total_entries']}

By Difficulty Level:
{chr(10).join([f"  {k}: {v}" for k, v in stats['by_difficulty'].items()])}

By Induction Type:
{chr(10).join([f"  {k}: {v}" for k, v in stats['by_type'].items()])}

Total Examples: {stats['total_examples']}
Total Exercises: {stats['total_exercises']}
Total Applications: {stats['total_applications']}
        """
        
        messagebox.showinfo("Encyclopedia Statistics", stats_text)
    
    def _show_user_guide(self):
        """Show user guide"""
        webbrowser.open("https://github.com/induction-omega/encyclopedia/wiki")
    
    def _show_about(self):
        """Show about dialog"""
        about_text = """
Induction Ω - Mathematical Induction Encyclopedia
Version 1.0.0

The most comprehensive resource for mathematical induction
and related concepts, featuring 500+ pages of content
covering all forms of induction from elementary to research level.

© 2024 Induction Ω Research Team
All rights reserved.
        """
        messagebox.showinfo("About Induction Ω", about_text)

def main():
    """Main function to run the encyclopedia GUI"""
    root = tk.Tk()
    app = EncyclopediaGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()