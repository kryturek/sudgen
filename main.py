import random
import copy
from fastapi import FastAPI, HTTPException, Depends, Response, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime, timedelta, timezone
import jwt as pyjwt  # Rename import to avoid confusion
import bcrypt
from typing import Optional, List
import json
import os
from dotenv import load_dotenv
from pathlib import Path
import asyncpg
from contextlib import asynccontextmanager
import logging
from fastapi.middleware.trustedhost import TrustedHostMiddleware

# Get absolute path to .env file
ENV_PATH = Path(__file__).resolve().parent / '.env'

# Try to load from .env file if it exists, otherwise use environment variables
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH, override=True)
    print(f"Loaded environment from .env file at: {ENV_PATH}")
else:
    print("No .env file found, using environment variables")

# Get required environment variables
db_url = os.getenv('DATABASE_URL')
jwt_secret = os.getenv('JWT_SECRET')

if not db_url or not jwt_secret:
    raise ValueError(
        "Required environment variables DATABASE_URL and JWT_SECRET must be set "
        "either in .env file or as environment variables"
    )

print(f"Database URL found with length: {len(db_url)}")

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
FRONTEND_URL = os.getenv('FRONTEND_URL', 'http://localhost:5173')

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
                initial_puzzle JSONB NOT NULL,     -- Starting puzzle state
                current_puzzle JSONB NOT NULL,     -- Current progress
                solution JSONB NOT NULL,           -- Complete solution
                pencil_marks JSONB,
                difficulty INTEGER,
                completed BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
        ''')

pool = None

# Update lifespan to use environment variable
@asynccontextmanager
async def lifespan(app: FastAPI):
    global pool
    try:
        logger.info("Establishing database connection...")
        pool = await asyncpg.create_pool(
            dsn=db_url,
            min_size=1,
            max_size=20,  # Increased for production
            ssl='require' if ENVIRONMENT == 'production' else None,
            timeout=30,
            command_timeout=30,
            server_settings={
                'application_name': 'sudgen',
                'client_encoding': 'utf8'
            }
        )
        logger.info("Database pool created successfully")
        
        await init_db(pool)
        logger.info("Database initialized")
        yield
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise
    finally:
        if pool:
            await pool.close()
            logger.info("Database pool closed")

# Create FastAPI app with lifespan
app = FastAPI(lifespan=lifespan)

# Update CORS middleware with more specific settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Set-Cookie", "Access-Control-Allow-Headers", 
                  "Access-Control-Allow-Origin", "Authorization"],
    expose_headers=["Set-Cookie"]
)

# Add TrustedHost middleware for production
if ENVIRONMENT == 'production':
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=ALLOWED_HOSTS
    )

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
    game_id: Optional[int]      # Only needed for updates
    current_puzzle: List[List[int]]   
    pencilMarks: dict
    completed: bool

class LoginData(BaseModel):
    email: str
    password: str

# JWT settings
SECRET_KEY = os.getenv('JWT_SECRET')  # Get from environment instead of hardcoding
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 1 week

# Auth utilities
def create_access_token(data: dict):
    to_encode = data.copy()
    
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return pyjwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(request: Request) -> Optional[User]:
    try:
        token = request.cookies.get("session")
        if not token:
            return None

        payload = pyjwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if not payload or "sub" not in payload:
            return None

        async with pool.acquire() as conn:
            user = await conn.fetchrow('SELECT * FROM users WHERE id = $1', int(payload["sub"]))
            if not user:
                return None
                
            return User(
                id=user['id'],
                username=user['username'],
                email=user['email'],
                points=user['points'],
                games_played=user['games_played'],
                games_won=user['games_won']
            )
    except pyjwt.JWTError:
        return None
    except Exception as e:
        print(f"User auth error: {str(e)}")
        return None

# Add this helper function to serialize dates
def serialize_game(game):
    return {
        "id": game["id"],
        "puzzle": game["puzzle"],
        "solution": game["solution"],
        "pencil_marks": game["pencil_marks"],
        "difficulty": game["difficulty"],
        "started_at": game["started_at"].isoformat() if game["started_at"] else None,
        "last_played": game["last_played"].isoformat() if game["last_played"] else None,
        "completed": game["completed"]
    }

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
async def login(login_data: LoginData, response: Response = None):
    async with pool.acquire() as conn:
        user = await conn.fetchrow(
            'SELECT * FROM users WHERE email = $1',
            login_data.email
        )
        
        if not user or not bcrypt.checkpw(
            login_data.password.encode(),
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

# Update the get_session endpoint
@app.get("/auth/session")
async def get_session(current_user: User = Depends(get_current_user)):
    if not current_user:
        return JSONResponse({
            "isAuthenticated": False,
            "user": None,
            "savedGames": []
        })
    
    async with pool.acquire() as conn:
        saved_games = await conn.fetch(
            'SELECT id, puzzle, solution, pencil_marks, difficulty, started_at, last_played, completed FROM saved_games WHERE user_id = $1 ORDER BY last_played DESC',
            current_user.id
        )
    
    return JSONResponse({
        "isAuthenticated": True,
        "user": {
            "id": current_user.id,
            "username": current_user.username,
            "email": current_user.email,
            "points": current_user.points,
            "games_played": current_user.games_played,
            "games_won": current_user.games_won
        },
        "savedGames": [serialize_game(dict(game)) for game in saved_games]
    })

# Update the get_game endpoint
@app.get("/games/{game_id}")
async def get_game(game_id: int, current_user: User = Depends(get_current_user)):
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    async with pool.acquire() as conn:
        game = await conn.fetchrow(
            'SELECT * FROM saved_games WHERE id = $1 AND user_id = $2', 
            game_id, current_user.id
        )
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    
    return serialize_game(dict(game))

# Game endpoints
@app.post("/games/start")
async def start_game(puzzle_data: dict, current_user: User = Depends(get_current_user)):
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        async with pool.acquire() as conn:
            game_id = await conn.fetchval('''
                INSERT INTO saved_games 
                (user_id, initial_puzzle, current_puzzle, solution, difficulty, completed)
                VALUES ($1, $2, $3, $4, $5, false)
                RETURNING id
            ''', 
                current_user.id,
                json.dumps(puzzle_data["puzzle"]),
                json.dumps(puzzle_data["puzzle"]),  # Initial current_puzzle is same as initial
                json.dumps(puzzle_data["solution"]),
                puzzle_data["difficulty"]
            )
            return {"id": game_id}
    except Exception as e:
        print(f"Start game error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/games/{game_id}/update")
async def update_game(game_id: int, game: SavedGame, current_user: User = Depends(get_current_user)):
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        async with pool.acquire() as conn:
            # Verify game belongs to user
            existing = await conn.fetchrow(
                'SELECT id FROM saved_games WHERE id = $1 AND user_id = $2',
                game_id, current_user.id
            )
            if not existing:
                raise HTTPException(status_code=404, detail="Game not found")

            await conn.execute('''
                UPDATE saved_games 
                SET current_puzzle = $1, pencil_marks = $2, completed = $3
                WHERE id = $4 AND user_id = $5
            ''', 
                json.dumps(game.current_puzzle),
                json.dumps(game.pencilMarks),
                game.completed,
                game_id,
                current_user.id
            )
            
            return {"message": "Game updated successfully"}
    except Exception as e:
        print(f"Update game error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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