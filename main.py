import random
import copy
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()

def generateBoard():
    board = []
    for i in range(9):
        board.append([0] * 9)
    return board

def printBoard(board):
    for i in range(9):
        for j in range(9):
            print(board[i][j], end=' ')
        print()

def findEmpty(board):
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                return (i, j)
    return None

def validateNumber(board, row, col, num):
    # Check row
    if num in board[row]:
        return False

    # Check column
    for i in range(9):
        if board[i][col] == num:
            return False

    # Check 3x3 grid
    startRow = row - row % 3
    startCol = col - col % 3
    for i in range(3):
        for j in range(3):
            if board[i + startRow][j + startCol] == num:
                return False

    return True

def fillBoard(board):
    empty = findEmpty(board)
    if not empty:
        return True  # Board filled successfully
    row, col = empty

    numbers = list(range(1, 10))
    random.shuffle(numbers)
    for num in numbers:
        if validateNumber(board, row, col, num):
            board[row][col] = num
            if fillBoard(board):
                return True
            board[row][col] = 0  # Reset on backtrack

    return False

def countSolutions(board):
    """Recursively count the number of solutions for the board."""
    empty = findEmpty(board)
    if not empty:
        return 1  # Found one complete solution
    row, col = empty
    count = 0
    for num in range(1, 10):
        if validateNumber(board, row, col, num):
            board[row][col] = num
            count += countSolutions(board)
            # Early exit if more than one solution found
            if count > 1:
                board[row][col] = 0
                return count
            board[row][col] = 0
    return count

def removeNumbersUnique(board, removals):
    """Removes numbers from the board while ensuring a unique solution.
       This version shuffles cell coordinates once rather than picking
       random indices repeatedly.
    """
    # get all cell positions in a list and shuffle once
    positions = [(i, j) for i in range(9) for j in range(9)]
    random.shuffle(positions)
    removed = 0

    # iterate through each candidate cell
    for i, j in positions:
        if removed >= removals:
            break
        if board[i][j] != 0:
            backup = board[i][j]
            board[i][j] = 0

            # Check uniqueness using a deep copy
            boardCopy = copy.deepcopy(board)
            solCount = countSolutions(boardCopy)

            if solCount != 1:
                # Restore if puzzle is no longer unique
                board[i][j] = backup
            else:
                removed += 1

    return board

@app.get("/sudoku")
def get_sudoku(removals: int = 50):
    """
    Generate a Sudoku puzzle with a unique solution.
    
    Query parameter: removals (default 50) is the number of cells to remove.
    Returns the puzzle (with removals), the complete solution, and the number of removed cells.
    """
    board = generateBoard()
    if fillBoard(board):
        # Save a deep copy of the solved board as the solution
        solution = copy.deepcopy(board)
        
        # Remove numbers to generate the puzzle
        puzzle = removeNumbersUnique(board, removals)
        
        # Count how many cells have been removed (i.e. are 0)
        removed_count = sum(cell == 0 for row in puzzle for cell in row)
        
        return {"puzzle": puzzle, "solution": solution, "removed_count": removed_count}
    else:
        return JSONResponse(content={"error": "Could not generate a valid board"}, status_code=500)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)