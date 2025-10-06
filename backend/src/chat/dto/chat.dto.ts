import { IsString, IsOptional, IsEnum } from 'class-validator';
import { MessageRole } from '@prisma/client';

export class CreateThreadDto {
  @IsOptional()
  @IsString()
  title?: string;
}

export class SendMessageDto {
  @IsString()
  content: string;

  @IsEnum(MessageRole)
  role: MessageRole;

  @IsOptional()
  @IsString()
  reasoning?: string;

  // Enhanced AI tracking fields
  @IsOptional()
  confidence?: number;

  @IsOptional()
  processingTimeMs?: number;

  @IsOptional()
  @IsString()
  analysisType?: string;

  @IsOptional()
  @IsString()
  stockSymbols?: string; // JSON string

  @IsOptional()
  @IsString()
  financialData?: string; // JSON string

  @IsOptional()
  @IsString()
  insights?: string; // JSON string

  @IsOptional()
  webSearchUsed?: boolean;

  @IsOptional()
  realTimeData?: boolean;

  @IsOptional()
  sourcesCount?: number;
}