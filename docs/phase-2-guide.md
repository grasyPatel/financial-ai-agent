# Phase 2: Database & Authentication

## 🎯 Goals for Phase 2
- Set up PostgreSQL database connection
- Implement user registration and login
- Add JWT authentication system
- Create protected API routes
- Connect frontend to authentication flow

## 📚 Key Concepts You'll Learn

### 1. Database Relationships
Understanding how data connects:
- **Users** can have many **Sessions** (one-to-many)
- **Users** can have many **Threads** (one-to-many)  
- **Threads** can have many **Messages** (one-to-many)
- **Messages** can reference many **Sources** (many-to-many)

### 2. Authentication vs Authorization
- **Authentication**: "Who are you?" (login process)
- **Authorization**: "What can you access?" (permissions)

### 3. JWT (JSON Web Tokens)
Think of JWT like a digital ID card:
- Contains user information
- Signed by the server (can't be faked)
- Expires after a set time
- Sent with every request to prove identity

### 4. Password Security
- **Never store plain text passwords!**
- **Hashing**: One-way transformation (can't be reversed)
- **Salt**: Random data added before hashing (prevents rainbow table attacks)
- **bcrypt**: Industry-standard password hashing library

### 5. Database Migrations
- **Migration**: A script that changes database structure
- **Up migration**: Apply changes (create table, add column)
- **Down migration**: Reverse changes (drop table, remove column)
- **Version control for your database schema**

## 🛠️ Step-by-Step Implementation

### Step 2.1: Set Up Prisma Database Connection

**What we're doing**: Connect our NestJS backend to PostgreSQL using Prisma ORM.

**Key files to create/modify**:
- `backend/src/prisma/prisma.service.ts` - Database connection service
- `backend/src/app.module.ts` - Add Prisma to main app module
- Run database migrations

### Step 2.2: Create User Registration

**What we're doing**: Allow new users to create accounts with email/password.

**Key concepts**:
- Input validation (email format, password strength)
- Password hashing with bcrypt
- Duplicate email checking
- Return user data (without password!)

**API Endpoint**: `POST /auth/register`
```json
{
  "email": "user@example.com",
  "password": "SecurePass123",
  "name": "John Doe"
}
```

### Step 2.3: Create User Login

**What we're doing**: Authenticate existing users and return JWT token.

**Key concepts**:
- Email/password verification
- Password comparison (bcrypt.compare)
- JWT token generation
- Session creation in database

**API Endpoint**: `POST /auth/login`
```json
{
  "email": "user@example.com", 
  "password": "SecurePass123"
}
```

**Response**:
```json
{
  "user": {
    "id": "cuid123",
    "email": "user@example.com",
    "name": "John Doe"
  },
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

### Step 2.4: Create Protected Routes

**What we're doing**: Add middleware to protect certain API endpoints.

**Key concepts**:
- **Guards**: NestJS middleware that runs before route handlers
- **JWT Strategy**: Validates JWT tokens
- **Request decoration**: Add user info to request object
- **Route protection**: Only authenticated users can access

### Step 2.5: Frontend Authentication Flow

**What we're doing**: Add login/register forms and store authentication state.

**Key concepts**:
- **React state management**: Store user and token
- **Local storage**: Persist authentication across browser sessions
- **Axios interceptors**: Automatically add auth headers
- **Protected routes**: Redirect to login if not authenticated

## 🎯 Phase 2 Deliverables

By the end of this phase, you'll have:

1. ✅ **Database Connection**: PostgreSQL connected via Prisma
2. ✅ **User Registration**: New users can create accounts
3. ✅ **User Login**: Existing users can sign in
4. ✅ **JWT Authentication**: Secure token-based auth system
5. ✅ **Protected APIs**: Routes that require authentication
6. ✅ **Frontend Auth**: Login/register forms with state management
7. ✅ **Session Persistence**: Users stay logged in across browser sessions
8. ✅ **Security Best Practices**: Hashed passwords, secure tokens

## 🧪 Testing Phase 2

We'll know Phase 2 is complete when:
- ✅ User can register with email/password
- ✅ User can login and receive JWT token
- ✅ Protected routes return 401 for unauthenticated requests
- ✅ Frontend shows different UI for logged-in vs logged-out users
- ✅ User sessions persist across browser refresh
- ✅ Database contains users, sessions, and threads tables

## 📖 Database Schema Deep Dive

Let's understand our Prisma schema:

```prisma
// User table - core user information
model User {
  id        String   @id @default(cuid())  // Unique ID
  email     String   @unique               // Login identifier  
  password  String                         // Hashed password
  name      String?                        // Optional display name
  createdAt DateTime @default(now())       // When account was created
  updatedAt DateTime @updatedAt            // Last profile update

  // Relationships (one user has many...)
  sessions Session[]  // Active login sessions
  threads  Thread[]   // Chat conversations
}

// Session table - tracks user logins
model Session {
  id        String   @id @default(cuid())
  userId    String                         // Which user owns this session
  token     String   @unique               // JWT token value
  expiresAt DateTime                       // When token expires
  createdAt DateTime @default(now())

  // Relationship (session belongs to one user)
  user User @relation(fields: [userId], references: [id], onDelete: Cascade)
}
```

## 🔐 Security Considerations

### Password Security
```typescript
// NEVER do this:
const user = { password: "plaintext123" }  // ❌ DANGER!

// Always do this:
const saltRounds = 10;
const hashedPassword = await bcrypt.hash(password, saltRounds);  // ✅ SAFE
```

### JWT Best Practices
- **Short expiration**: 1-7 days maximum
- **Strong secret**: Random, long, environment-specific  
- **HTTPS only**: Never send tokens over HTTP
- **Secure storage**: HttpOnly cookies > localStorage

### Input Validation
```typescript
// Validate email format
@IsEmail()
email: string;

// Validate password strength  
@MinLength(8)
@Matches(/^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)/)
password: string;
```

## 🚨 Common Beginner Pitfalls

1. **Storing passwords in plain text**: Always hash passwords!
2. **Weak JWT secrets**: Use long, random secrets
3. **No input validation**: Validate all user inputs
4. **Exposing passwords in responses**: Never return password field
5. **No error handling**: Always handle database and network errors
6. **Hardcoded credentials**: Use environment variables

## 🧩 Architecture Overview for Phase 2

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│    Frontend     │    │    Backend      │    │   Database      │
│                 │    │                 │    │                 │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │Login Form   │ │───▶│ │Auth Routes  │ │───▶│ │Users Table  │ │
│ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │
│                 │    │                 │    │                 │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │Chat UI      │ │───▶│ │JWT Guard    │ │───▶│ │Sessions     │ │
│ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 💡 Learning Checkpoints

Before starting each step, make sure you understand:

**Step 2.1**: What is an ORM? How does Prisma work?
**Step 2.2**: How does password hashing work? Why use bcrypt?
**Step 2.3**: What's inside a JWT token? How is it verified?
**Step 2.4**: What's the difference between middleware and guards?
**Step 2.5**: How does React manage authentication state?

## 🎉 Ready to Start Phase 2?

**Prerequisites check**:
- ✅ Phase 1 complete (all services running)
- ✅ Docker and Docker Compose working
- ✅ PostgreSQL container accessible
- ✅ Understanding of basic authentication concepts

**Time estimate**: 4-6 hours for beginners

**What you'll build**: A complete authentication system from scratch!

Let me know when you're ready to start implementing! 🚀