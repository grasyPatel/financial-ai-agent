import { Injectable, Logger } from '@nestjs/common';

@Injectable()
export class AiService {
  private readonly logger = new Logger(AiService.name);
  private readonly aiServiceUrl = process.env.AI_SERVICE_URL || 'http://localhost:9000';

  /**
   * Send a message to the AI service and get analysis
   */
  async getFinancialAnalysis(message: string, sessionId: string): Promise<AiResponse> {
    try {
      const response = await fetch(`${this.aiServiceUrl}/research/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: message,
          session_id: sessionId,
        }),
      });

      if (!response.ok) {
        throw new Error(`AI service responded with status: ${response.status}`);
      }

      // Since it's a streaming response, we need to read the full stream
      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      let result = '';
      let finalResult: any = null;

      if (!reader) {
        throw new Error('No response body from AI service');
      }

      // Read the streaming response
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        result += chunk;

        // Look for final_report in the chunk
        const lines = chunk.split('\n');
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.substring(6));
              if (data.type === 'final_report') {
                finalResult = data;
              }
            } catch (e) {
              // Ignore parsing errors for non-JSON lines
            }
          }
        }
      }

      if (!finalResult) {
        throw new Error('No final report received from AI service');
      }

      // Extract analysis data from the AI response
      return this.parseAiResponse(finalResult);
    } catch (error) {
      this.logger.error(`Failed to get AI analysis: ${error.message}`, error.stack);
      throw error;
    }
  }

  /**
   * Parse the AI service response into structured data
   */
  private parseAiResponse(aiResponse: any): AiResponse {
    const stockSymbols = this.extractStockSymbols(aiResponse.financial_data || []);
    
    return {
      content: aiResponse.content || '',
      analysisType: this.determineAnalysisType(aiResponse.content),
      stockSymbols: stockSymbols.length > 0 ? JSON.stringify(stockSymbols) : null,
      financialData: aiResponse.financial_data ? JSON.stringify(aiResponse.financial_data) : null,
      insights: aiResponse.insights ? JSON.stringify(aiResponse.insights) : null,
      webSearchUsed: true, // AI service uses web search
      realTimeData: true, // AI service fetches real-time data
      sourcesCount: aiResponse.sources?.length || 0,
      sources: aiResponse.sources || [],
      confidence: aiResponse.confidence_score || 0.8,
      processingTimeMs: aiResponse.processing_time_ms || null,
    };
  }

  /**
   * Extract stock symbols from financial data
   */
  private extractStockSymbols(financialData: any[]): string[] {
    const symbols = new Set<string>();
    
    for (const data of financialData) {
      if (data.symbol) {
        symbols.add(data.symbol);
      }
    }
    
    return Array.from(symbols);
  }

  /**
   * Determine the type of analysis based on content
   */
  private determineAnalysisType(content: string): string {
    const contentLower = content.toLowerCase();
    
    if (contentLower.includes('comparison') || contentLower.includes('vs') || contentLower.includes('peer')) {
      return 'peer_comparison';
    }
    
    if (contentLower.includes('sector') || contentLower.includes('industry')) {
      return 'sector_analysis';
    }
    
    if (contentLower.includes('technical') || contentLower.includes('chart') || contentLower.includes('rsi')) {
      return 'technical_analysis';
    }
    
    if (contentLower.includes('fundamental') || contentLower.includes('earnings') || contentLower.includes('revenue')) {
      return 'fundamental_analysis';
    }
    
    // Check if it mentions specific stock symbols
    const hasStockSymbols = /\b[A-Z]{1,5}\b/.test(content);
    if (hasStockSymbols) {
      return 'stock_analysis';
    }
    
    return 'general_analysis';
  }
}

/**
 * Interface for AI service response
 */
export interface AiResponse {
  content: string;
  analysisType: string;
  stockSymbols: string | null;
  financialData: string | null;
  insights: string | null;
  webSearchUsed: boolean;
  realTimeData: boolean;
  sourcesCount: number;
  sources: any[];
  confidence: number;
  processingTimeMs: number | null;
}