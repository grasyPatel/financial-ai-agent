import { Injectable } from '@nestjs/common';

@Injectable()
export class AppService {
  getHello(): string {
    return '🤖 Deep Finance Research Chatbot Backend is running! Phase 1 Complete ✅';
  }
}
