import { Injectable, NotFoundException, ForbiddenException } from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service';
import { CreateThreadDto, SendMessageDto } from './dto/chat.dto';
import { AiService } from '../services/ai.service';

@Injectable()
export class ChatService {
  constructor(
    private prisma: PrismaService,
    private aiService: AiService,
  ) {}

  async createThread(userId: string, createThreadDto: CreateThreadDto) {
    const thread = await this.prisma.thread.create({
      data: {
        userId,
        title: createThreadDto.title || 'New Chat',
      },
      include: {
        messages: {
          orderBy: { createdAt: 'asc' },
          take: 10,
        },
        _count: {
          select: { messages: true },
        },
      },
    });

    return thread;
  }

  async getThreads(userId: string, limit: number = 20) {
    const threads = await this.prisma.thread.findMany({
      where: { userId },
      orderBy: { updatedAt: 'desc' },
      take: limit,
      include: {
        messages: {
          orderBy: { createdAt: 'desc' },
          take: 1, // Get the last message for preview
        },
        _count: {
          select: { messages: true },
        },
      },
    });

    return threads.map(thread => ({
      ...thread,
      lastMessage: thread.messages[0] || null,
      messageCount: thread._count.messages,
    }));
  }

  async getThread(userId: string, threadId: string) {
    const thread = await this.prisma.thread.findFirst({
      where: { id: threadId, userId },
      include: {
        messages: {
          orderBy: { createdAt: 'asc' },
          include: {
            sources: true,
          },
        },
        _count: {
          select: { messages: true },
        },
      },
    });

    if (!thread) {
      throw new NotFoundException('Thread not found');
    }

    return thread;
  }

  async getMessages(
    userId: string,
    threadId: string,
    limit: number = 50,
    offset: number = 0,
  ) {
    // Verify thread ownership
    const thread = await this.prisma.thread.findFirst({
      where: { id: threadId, userId },
    });

    if (!thread) {
      throw new NotFoundException('Thread not found');
    }

    const messages = await this.prisma.message.findMany({
      where: { threadId },
      orderBy: { createdAt: 'asc' },
      take: limit,
      skip: offset,
      include: {
        sources: true,
      },
    });

    const total = await this.prisma.message.count({
      where: { threadId },
    });

    return {
      messages,
      pagination: {
        total,
        limit,
        offset,
        hasMore: offset + limit < total,
      },
    };
  }

  async saveMessage(userId: string, threadId: string, messageDto: SendMessageDto) {
    // Verify thread ownership
    const thread = await this.prisma.thread.findFirst({
      where: { id: threadId, userId },
    });

    if (!thread) {
      throw new NotFoundException('Thread not found');
    }

    // First, save the user's message
    const userMessage = await this.prisma.message.create({
      data: {
        threadId,
        content: messageDto.content,
        role: messageDto.role,
        reasoning: messageDto.reasoning,
      },
      include: {
        sources: true,
      },
    });

    // If this is a user message, get AI analysis and create assistant response
    if (messageDto.role === 'USER') {
      try {
        const aiResponse = await this.aiService.getFinancialAnalysis(
          messageDto.content,
          `${userId}-${threadId}`,
        );

        // Create AI assistant response with enhanced fields
        const assistantMessage = await this.prisma.message.create({
          data: {
            threadId,
            content: aiResponse.content,
            role: 'ASSISTANT',
            reasoning: `AI-generated financial analysis (${aiResponse.analysisType}) - Sources: ${aiResponse.sourcesCount}`,
            // Enhanced fields from AI analysis
            confidence: aiResponse.confidence,
            processingTimeMs: aiResponse.processingTimeMs,
            analysisType: aiResponse.analysisType,
            stockSymbols: aiResponse.stockSymbols ? JSON.stringify(aiResponse.stockSymbols) : null,
            financialData: aiResponse.financialData ? JSON.stringify(aiResponse.financialData) : null,
            insights: aiResponse.insights ? JSON.stringify(aiResponse.insights) : null,
            webSearchUsed: aiResponse.webSearchUsed || false,
            realTimeData: aiResponse.realTimeData || true,
            sourcesCount: aiResponse.sourcesCount || 0,
          },
          include: {
            sources: true,
          },
        });

        // Create source records for the AI response
        if (aiResponse.sources && aiResponse.sources.length > 0) {
          await this.createSourcesForMessage(assistantMessage.id, aiResponse.sources);
        }

        // Update thread's updatedAt timestamp
        await this.prisma.thread.update({
          where: { id: threadId },
          data: { updatedAt: new Date() },
        });

        // Return both messages
        return {
          userMessage,
          assistantMessage: {
            ...assistantMessage,
            sources: await this.prisma.source.findMany({
              where: { messages: { some: { id: assistantMessage.id } } },
            }),
          },
        };
      } catch (error) {
        console.error('AI service error:', error);
        
        // If AI service fails, still save user message but create error response
        const errorMessage = await this.prisma.message.create({
          data: {
            threadId,
            content: 'I apologize, but I encountered an error processing your request. Please try again.',
            role: 'ASSISTANT',
            reasoning: 'Error response due to AI service failure',
          },
        });

        await this.prisma.thread.update({
          where: { id: threadId },
          data: { updatedAt: new Date() },
        });

        return {
          userMessage,
          assistantMessage: errorMessage,
        };
      }
    }

    // For assistant messages, just save as-is
    await this.prisma.thread.update({
      where: { id: threadId },
      data: { updatedAt: new Date() },
    });

    return { userMessage };
  }

  async getMessage(userId: string, threadId: string, messageId: string) {
    // Verify thread ownership
    const thread = await this.prisma.thread.findFirst({
      where: { id: threadId, userId },
    });

    if (!thread) {
      throw new NotFoundException('Thread not found');
    }

    const message = await this.prisma.message.findFirst({
      where: { id: messageId, threadId },
      include: {
        sources: true,
      },
    });

    if (!message) {
      throw new NotFoundException('Message not found');
    }

    return message;
  }

  async updateThreadTitle(userId: string, threadId: string, title: string) {
    const thread = await this.prisma.thread.findFirst({
      where: { id: threadId, userId },
    });

    if (!thread) {
      throw new NotFoundException('Thread not found');
    }

    return this.prisma.thread.update({
      where: { id: threadId },
      data: { title },
    });
  }

  async deleteThread(userId: string, threadId: string) {
    const thread = await this.prisma.thread.findFirst({
      where: { id: threadId, userId },
    });

    if (!thread) {
      throw new NotFoundException('Thread not found');
    }

    await this.prisma.thread.delete({
      where: { id: threadId },
    });

    return { message: 'Thread deleted successfully' };
  }

  async createSourcesForMessage(messageId: string, sources: any[]) {
    /**
     * Create enhanced source records for a message
     * @param messageId - The message ID to associate sources with
     * @param sources - Array of source data from AI analysis
     */
    if (!sources || sources.length === 0) return [];

    const createdSources: any[] = [];

    for (const source of sources) {
      try {
        const sourceData = {
          title: source.title || source.name || 'Financial Data Source',
          url: source.url || null,
          snippet: source.snippet || source.description || null,
          domain: source.domain || this.extractDomain(source.url),
          sourceType: this.mapSourceType(source),
          reliability: source.reliability || 7, // Default good reliability
          dataType: source.dataType || this.inferDataType(source),
          ticker: source.ticker || source.symbol || null,
          exchange: source.exchange || null,
          sector: source.sector || null,
          apiSource: source.apiSource || 'Yahoo Finance',
          fetchedAt: new Date(),
          isRealTime: source.isRealTime || true,
          confidence: source.confidence || 0.8,
        };

        const createdSource = await this.prisma.source.create({
          data: {
            ...sourceData,
            messages: {
              connect: { id: messageId }
            }
          },
        });

        createdSources.push(createdSource);
      } catch (error) {
        console.error('Error creating source:', error);
        // Continue with other sources even if one fails
      }
    }

    return createdSources;
  }

  private extractDomain(url: string): string | null {
    if (!url) return null;
    try {
      return new URL(url).hostname;
    } catch {
      return null;
    }
  }

  private mapSourceType(source: any) {
    // Map source information to our SourceType enum values
    const apiSource = source.apiSource?.toLowerCase() || '';
    const url = source.url?.toLowerCase() || '';
    const title = source.title?.toLowerCase() || '';

    if (apiSource.includes('yahoo') || url.includes('yahoo')) return 'YAHOO_FINANCE' as any;
    if (apiSource.includes('sec') || url.includes('sec.gov')) return 'SEC_FILING' as any;
    if (apiSource.includes('bloomberg')) return 'BLOOMBERG' as any;
    if (apiSource.includes('reuters')) return 'REUTERS' as any;
    if (title.includes('earnings')) return 'EARNINGS_CALL' as any;
    if (title.includes('analyst')) return 'ANALYST_REPORT' as any;
    if (source.isRealTime) return 'REAL_TIME_QUOTE' as any;
    if (source.dataType === 'technical') return 'TECHNICAL_INDICATOR' as any;
    
    return 'FINANCIAL_NEWS' as any; // Default fallback
  }

  private inferDataType(source: any): string {
    const title = source.title?.toLowerCase() || '';
    const content = source.snippet?.toLowerCase() || '';
    
    if (title.includes('price') || title.includes('quote')) return 'stock_price';
    if (title.includes('earnings') || content.includes('earnings')) return 'earnings';
    if (title.includes('news') || title.includes('announcement')) return 'news';
    if (title.includes('technical') || title.includes('indicator')) return 'technical_analysis';
    if (title.includes('comparison') || title.includes('peer')) return 'peer_comparison';
    
    return 'general'; // Default
  }
}