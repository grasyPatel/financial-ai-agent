import {
  Controller,
  Get,
  Post,
  Body,
  Param,
  Query,
  UseGuards,
  Request,
} from '@nestjs/common';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { ChatService } from './chat.service';
import { CreateThreadDto, SendMessageDto } from './dto/chat.dto';

@Controller('chat')
@UseGuards(JwtAuthGuard)
export class ChatController {
  constructor(private readonly chatService: ChatService) {}

  @Post('threads')
  async createThread(@Request() req: any, @Body() createThreadDto: CreateThreadDto) {
    return this.chatService.createThread(req.user.id, createThreadDto);
  }

  @Get('threads')
  async getThreads(@Request() req: any, @Query('limit') limit?: string) {
    const limitNum = limit ? parseInt(limit, 10) : 20;
    return this.chatService.getThreads(req.user.id, limitNum);
  }

  @Get('threads/:threadId')
  async getThread(@Request() req: any, @Param('threadId') threadId: string) {
    return this.chatService.getThread(req.user.id, threadId);
  }

  @Get('threads/:threadId/messages')
  async getMessages(
    @Request() req: any,
    @Param('threadId') threadId: string,
    @Query('limit') limit?: string,
    @Query('offset') offset?: string,
  ) {
    const limitNum = limit ? parseInt(limit, 10) : 50;
    const offsetNum = offset ? parseInt(offset, 10) : 0;
    return this.chatService.getMessages(req.user.id, threadId, limitNum, offsetNum);
  }

  @Post('threads/:threadId/messages')
  async sendMessage(
    @Request() req: any,
    @Param('threadId') threadId: string,
    @Body() sendMessageDto: SendMessageDto,
  ) {
    return this.chatService.saveMessage(req.user.id, threadId, sendMessageDto);
  }

  @Get('threads/:threadId/messages/:messageId')
  async getMessage(
    @Request() req: any,
    @Param('threadId') threadId: string,
    @Param('messageId') messageId: string,
  ) {
    return this.chatService.getMessage(req.user.id, threadId, messageId);
  }
}