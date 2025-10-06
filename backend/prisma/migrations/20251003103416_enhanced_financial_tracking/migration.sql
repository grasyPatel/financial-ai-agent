/*
  Warnings:

  - Added the required column `sourceType` to the `sources` table without a default value. This is not possible if the table is not empty.

*/
-- CreateTable
CREATE TABLE "analysis_sessions" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "userId" TEXT NOT NULL,
    "query" TEXT NOT NULL,
    "sessionId" TEXT NOT NULL,
    "analysisType" TEXT NOT NULL,
    "symbolsAnalyzed" TEXT NOT NULL,
    "totalProcessingMs" INTEGER,
    "sourcesUsed" INTEGER,
    "confidenceScore" REAL,
    "keyFindings" TEXT,
    "recommendations" TEXT,
    "riskAssessment" TEXT,
    "ipAddress" TEXT,
    "userAgent" TEXT,
    "apiVersion" TEXT,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "completedAt" DATETIME,
    CONSTRAINT "analysis_sessions_userId_fkey" FOREIGN KEY ("userId") REFERENCES "users" ("id") ON DELETE CASCADE ON UPDATE CASCADE
);

-- RedefineTables
PRAGMA defer_foreign_keys=ON;
PRAGMA foreign_keys=OFF;
CREATE TABLE "new_messages" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "threadId" TEXT NOT NULL,
    "content" TEXT NOT NULL,
    "role" TEXT NOT NULL,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "reasoning" TEXT,
    "confidence" REAL,
    "processingTimeMs" INTEGER,
    "analysisType" TEXT,
    "stockSymbols" TEXT,
    "financialData" TEXT,
    "insights" TEXT,
    "webSearchUsed" BOOLEAN NOT NULL DEFAULT false,
    "realTimeData" BOOLEAN NOT NULL DEFAULT false,
    "sourcesCount" INTEGER,
    CONSTRAINT "messages_threadId_fkey" FOREIGN KEY ("threadId") REFERENCES "threads" ("id") ON DELETE CASCADE ON UPDATE CASCADE
);
INSERT INTO "new_messages" ("content", "createdAt", "id", "reasoning", "role", "threadId") SELECT "content", "createdAt", "id", "reasoning", "role", "threadId" FROM "messages";
DROP TABLE "messages";
ALTER TABLE "new_messages" RENAME TO "messages";
CREATE TABLE "new_sources" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "url" TEXT,
    "title" TEXT NOT NULL,
    "snippet" TEXT,
    "domain" TEXT,
    "publishedAt" DATETIME,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "sourceType" TEXT NOT NULL,
    "reliability" INTEGER NOT NULL DEFAULT 5,
    "dataType" TEXT,
    "ticker" TEXT,
    "exchange" TEXT,
    "marketCap" REAL,
    "sector" TEXT,
    "apiSource" TEXT,
    "fetchedAt" DATETIME,
    "isRealTime" BOOLEAN NOT NULL DEFAULT false,
    "confidence" REAL
);
INSERT INTO "new_sources" ("createdAt", "domain", "id", "publishedAt", "snippet", "title", "url") SELECT "createdAt", "domain", "id", "publishedAt", "snippet", "title", "url" FROM "sources";
DROP TABLE "sources";
ALTER TABLE "new_sources" RENAME TO "sources";
CREATE TABLE "new_users" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "email" TEXT NOT NULL,
    "password" TEXT NOT NULL,
    "name" TEXT,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" DATETIME NOT NULL,
    "totalQueries" INTEGER NOT NULL DEFAULT 0,
    "lastActiveAt" DATETIME,
    "subscriptionTier" TEXT
);
INSERT INTO "new_users" ("createdAt", "email", "id", "name", "password", "updatedAt") SELECT "createdAt", "email", "id", "name", "password", "updatedAt" FROM "users";
DROP TABLE "users";
ALTER TABLE "new_users" RENAME TO "users";
CREATE UNIQUE INDEX "users_email_key" ON "users"("email");
PRAGMA foreign_keys=ON;
PRAGMA defer_foreign_keys=OFF;
