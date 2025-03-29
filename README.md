# Shop Navigation System – 

This repository contains a comprehensive **Shop Finding and Navigation System**, developed as part of the Data Structures and Algorithms. The system allows users to add, remove, and update shops, create connections between them, find paths using DFS and BFS, search shops by category, and sort them by ratings — all via a Python-based interactive menu.

The application simulates real-world indoor navigation systems used in malls, airports, or commercial centers, and integrates key data structure concepts including:
- Graphs
- Hash Tables
- Max Heaps
- Linked Lists, Stacks, and Queues

---

## Objectives
- Represent shops as nodes and pathways as edges using a graph.
- Enable graph traversal (DFS and BFS) to find and compare paths.
- Store and search shop info using a hash table.
- Sort shops by rating using a max heap.
- Load and process data from Excel files dynamically.

---

## Features
### 🏪 Shop Management
- Add, delete, or update shop info (name, category, location, rating)
- Store each shop as a vertex in a graph structure

### 🔗 Pathway Operations
- Create or remove pathways (edges) between shops
- Display graph as adjacency list or matrix

### 🧭 Pathfinding & Navigation
- Use **DFS and BFS** to find paths between two shops
- Compare paths and determine the shortest

### 🔍 Category-Based Search
- Use a **hash table** for fast lookup of shops by category
- Case-insensitive searching with error handling

### ⭐ Rating-Based Sorting
- Sort shops within a category by rating using a **max heap**
- Display top-rated shops in descending order

### 📂 Excel File Loader
- Load shop and connection data from `test_data.xlsx`

---

## File Structure
| File | Description |
|------|-------------|
| `finalcode.py` | Main implementation of the system (includes all classes and logic) |
| `test_data.xlsx` | Excel file with two sheets: Shop data and Connections |
| `Project Requirements.pdf` | Assignment brief with specifications |
| `README` | Original README placeholder (replaced by this one) |

---

## How to Run
```bash
python finalcode.py
```

You’ll be prompted with an interactive menu to:
- Add/remove shops
- Add/remove edges
- Search categories
- Run DFS/BFS pathfinding
- Load data from Excel
- Sort shops by rating

Make sure `test_data.xlsx` is in the same directory when loading from file.

---

## Technologies Used
- `Python 3`
- `pandas` (for Excel data reading)
- `numpy`

Install required packages:
```bash
pip install pandas numpy
```

---

## Author
**Syed Muhammad Ahmed Zaidi**  

---

## License
This project is part of academic coursework and is intended for educational demonstration purposes only.
