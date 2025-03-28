import random
import copy
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
from pathlib import Path
import logging

# Get absolute path to .env file
ENV_PATH = Path(__file__).resolve().parent / '.env'

# Try to load from .env file if it exists, otherwise use environment variables
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH, override=True)
    print(f"Loaded environment from .env file at: {ENV_PATH}")
else:
    print("No .env file found, using environment variables")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Environment variables with defaults
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
PORT = int(os.getenv('PORT', 8000))
ALLOWED_HOSTS = os.getenv('ALLOWED_HOSTS', 'localhost').split(',')
FRONTEND_URL = os.getenv('FRONTEND_URL', 'https://kryturek.github.io')

def generateBoard():
    return [[0] * 9 for _ in range(9)]

def printBoard(board):
    for row in board:
        print(' '.join(str(cell) if cell != 0 else '.' for cell in row))

def getCandidates(board, row, col):
    """Return a list of valid numbers for the cell (row, col)."""
    if board[row][col] != 0:
        return []
    candidates = set(range(1, 10))
    # Remove numbers from the row and column
    candidates -= set(board[row])
    candidates -= {board[i][col] for i in range(9)}
    # Remove numbers from the 3x3 block
    startRow, startCol = row - row % 3, col - col % 3
    for i in range(3):
        for j in range(3):
            candidates.discard(board[startRow+i][startCol+j])
    return list(candidates)

def findEmptyMRV(board):
    """
    Find the empty cell with the fewest candidates.
    Returns (row, col, candidates) or (None, None, None) if board is full.
    """
    best = None
    best_candidates = None
    min_count = 10  # More than the maximum number of candidates.
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                candidates = getCandidates(board, i, j)
                if len(candidates) < min_count:
                    best = (i, j)
                    best_candidates = candidates
                    min_count = len(candidates)
                    if min_count == 0:
                        return best[0], best[1], []  # Dead end
                    if min_count == 1:
                        return best[0], best[1], best_candidates
    if best is None:
        return None, None, None
    return best[0], best[1], best_candidates

def fillBoard(board):
    row, col, candidates = findEmptyMRV(board)
    if row is None:
        return True  # Board completely filled
    # Shuffle candidates to randomize board generation
    random.shuffle(candidates)
    for num in candidates:
        board[row][col] = num
        if fillBoard(board):
            return True
        board[row][col] = 0  # Undo move (backtrack)
    return False

def countSolutions(board):
    """
    Count solutions using MRV and early exit as soon as >1 solution is found.
    """
    row, col, candidates = findEmptyMRV(board)
    if row is None:
        return 1  # A complete valid solution
    count = 0
    for num in candidates:
        board[row][col] = num
        count += countSolutions(board)
        if count > 1:  # Early exit when more than one solution is found.
            board[row][col] = 0
            return count
        board[row][col] = 0  # Undo move
    return count

def removeNumbersUnique(board, removals):
    """
    Remove numbers while ensuring puzzle uniqueness.
    Instead of repeatedly picking random cells, 
    we shuffle all cell positions and then iterate.
    """
    positions = [(i, j) for i in range(9) for j in range(9)]
    random.shuffle(positions)
    removed = 0
    for i, j in positions:
        if removed >= removals:
            break
        if board[i][j] != 0:
            backup = board[i][j]
            board[i][j] = 0
            # Use countSolutions on a deep copy to prevent interfering with current board state.
            boardCopy = copy.deepcopy(board)
            if countSolutions(boardCopy) != 1:
                board[i][j] = backup  # Restore because uniqueness lost
            else:
                removed += 1
    return board

# Create FastAPI app
app = FastAPI()

# Update CORS middleware with more specific settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL, "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Access-Control-Allow-Headers", 
                  "Access-Control-Allow-Origin"],
)

@app.get("/")
def root():
    html_content = """
    <html>
        <head>
            <title>Sudoku API</title>
        </head>
        <body>
            <h1>Welcome to the Sudoku API!</h1>
            <p>Use <a href="/sudoku">/sudoku</a> to get a Sudoku puzzle with 30 removed cells.</p>
            <p>To customize the number of removed cells, use <code>/sudoku?removals=x</code>, where <code>x</code> is the number of cells to remove.</p>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)


@app.get("/sudoku")
def get_sudoku(removals: int = 30):
    """
    Generate a Sudoku puzzle with a unique solution.
    Query param 'removals' (default 30) is the number of cells to remove.
    Returns the puzzle, the complete solution, and the removed count.
    """
    board = generateBoard()
    if fillBoard(board):
        solution = copy.deepcopy(board)
        puzzle = removeNumbersUnique(board, removals)
        removed_count = sum(cell == 0 for row in puzzle for cell in row)
        return {"puzzle": puzzle, "solution": solution, "removed_count": removed_count}
    else:
        return JSONResponse(content={"error": "Could not generate a valid board"}, status_code=500)

if __name__ == '__main__':
    import uvicorn
    port = int(os.environ.get('PORT', 10000))
    print(f"Starting server on 0.0.0.0:{port}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        server_header=False,
        proxy_headers=True
    )