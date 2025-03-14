import random
import copy


def generateBoard():
	board = []
	for i in range(9):
		board.append([0]*9)
	return board

def printBoard(board):
	for i in range(9):
		for j in range(9):
			print(board[i][j], end=' ')
		print()

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
		

def main():
    board = generateBoard()
    if fillBoard(board):  # fill the board in place, expect True if solved
        printBoard(board)
    else:
        print("No solution found.")


if __name__ == '__main__':
	main()