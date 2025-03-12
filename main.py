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
	boardcopy = copy.deepcopy(board)
	for i in range(9):
		for j in range(9):
			number = random.randint(1, 9)
			while not validateNumber(boardcopy, i, j, number):
				number = random.randint(1, 9)
			boardcopy[i][j] = number
			print('number assigned:', number, 'at', i, j)
	return boardcopy

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
	board = fillBoard(board)
	printBoard(board)


if __name__ == '__main__':
	main()