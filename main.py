import random
import copy
from fastapi import FastAPI, HTTPException, Depends, Response, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime, timedelta
import jwt
import bcrypt
from typing import Optional, List
import json
import os
from dotenv import load_dotenv
import asyncpg

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

@app.get("/sudoku")
def get_sudoku(removals: int = 3):
    """
    Generate a Sudoku puzzle with a unique solution.
    Query param 'removals' (default 50) is the number of cells to remove.
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

# Database connection pool
async def create_pool():
    return await asyncpg.create_pool(
        os.getenv('DATABASE_URL'),
        min_size=1,
        max_size=10
    )

# Create tables if they don't exist
async def init_db(pool):
    async with pool.acquire() as conn:
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username VARCHAR(50) UNIQUE NOT NULL,
                email VARCHAR(255) UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                points INTEGER DEFAULT 0,
                games_played INTEGER DEFAULT 0,
                games_won INTEGER DEFAULT 0,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS saved_games (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id),
                puzzle JSONB NOT NULL,
                solution JSONB NOT NULL,
                pencil_marks JSONB,
                difficulty INTEGER,
                started_at TIMESTAMP WITH TIME ZONE,
                last_played TIMESTAMP WITH TIME ZONE,
                completed BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
        ''')

app = FastAPI()
pool = None

@app.on_event("startup")
async def startup():
    global pool
    pool = await create_pool()
    await init_db(pool)

@app.on_event("shutdown")
async def shutdown():
    if pool:
        await pool.close()

# Auth models
class UserCreate(BaseModel):
    username: str
    email: str
    password: str

class User(BaseModel):
    id: int
    username: str
    email: str
    points: int
    games_played: int
    games_won: int

class SavedGame(BaseModel):
    puzzle: List[List[int]]
    solution: List[List[int]]
    pencil_marks: dict
    difficulty: int
    started_at: datetime
    last_played: datetime
    completed: bool

# JWT settings
SECRET_KEY = "your-secret-key-here"  # Change this in production!
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 1 week

# Auth utilities
def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(request: Request) -> Optional[User]:
    token = request.cookies.get("session")
    if not token:
        return None
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        async with pool.acquire() as conn:
            user = await conn.fetchrow('SELECT * FROM users WHERE id = $1', int(payload["sub"]))
            if user:
                return User(
                    id=user['id'],
                    username=user['username'],
                    email=user['email'],
                    points=user['points'],
                    games_played=user['games_played'],
                    games_won=user['games_won']
                )
    except:
        return None
    return None

# Auth endpoints
@app.post("/auth/register")
async def register(user: UserCreate):
    async with pool.acquire() as conn:
        # Check if user exists
        existing = await conn.fetchrow(
            'SELECT id FROM users WHERE email = $1 OR username = $2',
            user.email, user.username
        )
        if existing:
            raise HTTPException(status_code=400, detail="User already exists")
        
        # Hash password and create user
        password_hash = bcrypt.hashpw(user.password.encode(), bcrypt.gensalt()).decode()
        await conn.execute('''
            INSERT INTO users (username, email, password_hash)
            VALUES ($1, $2, $3)
        ''', user.username, user.email, password_hash)
        
        return {"message": "User created successfully"}

@app.post("/auth/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends(), response: Response = None):
    async with pool.acquire() as conn:
        user = await conn.fetchrow(
            'SELECT * FROM users WHERE email = $1',
            form_data.username
        )
        
        if not user or not bcrypt.checkpw(
            form_data.password.encode(), 
            user['password_hash'].encode()
        ):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        # Get saved games
        saved_games = await conn.fetch(
            'SELECT * FROM saved_games WHERE user_id = $1',
            user['id']
        )
        
        # Create session token
        token = create_access_token({"sub": str(user['id'])})
        response.set_cookie(
            key="session",
            value=token,
            httponly=True,
            max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            samesite="lax"
        )
        
        return {
            "user": {
                "id": user['id'],
                "username": user['username'],
                "email": user['email'],
                "points": user['points'],
                "games_played": user['games_played'],
                "games_won": user['games_won']
            },
            "saved_games": [dict(game) for game in saved_games]
        }

@app.post("/auth/logout")
async def logout(response: Response):
    response.delete_cookie("session")
    return {"message": "Logged out successfully"}

@app.get("/auth/session")
async def get_session(current_user: User = Depends(get_current_user)):
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return {"user": current_user}

# Game endpoints
@app.post("/games/save")
async def save_game(game: SavedGame, current_user: User = Depends(get_current_user)):
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    async with pool.acquire() as conn:
        game_id = await conn.fetchval('''
            INSERT INTO saved_games 
            (user_id, puzzle, solution, pencil_marks, difficulty, started_at, last_played, completed)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            RETURNING id
        ''', 
            current_user.id,
            json.dumps(game.puzzle),
            json.dumps(game.solution),
            json.dumps(game.pencil_marks),
            game.difficulty,
            game.started_at,
            game.last_played,
            game.completed
        )
        
        if game.completed:
            points = 10 * game.difficulty
            await conn.execute('''
                UPDATE users 
                SET points = points + $1, games_played = games_played + 1, games_won = games_won + 1
                WHERE id = $2
            ''', points, current_user.id)
        
        return {"id": game_id, "message": "Game saved successfully"}

@app.get("/games/{game_id}")
async def get_game(game_id: int, current_user: User = Depends(get_current_user)):
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    async with pool.acquire() as conn:
        game = await conn.fetchrow('SELECT * FROM saved_games WHERE id = $1 AND user_id = $2', 
                                   game_id, current_user.id)
    
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    
    return {
        "id": game['id'],
        "puzzle": game['puzzle'],
        "solution": game['solution'],
        "pencil_marks": game['pencil_marks'],
        "difficulty": game['difficulty'],
        "started_at": game['started_at'],
        "last_played": game['last_played'],
        "completed": game['completed']
    }

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)