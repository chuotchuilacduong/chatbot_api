import { Controller, Post, Body } from '@nestjs/common';
import { ChatbotService } from './chatbot.service';

@Controller('chatbot')
export class ChatbotController {
  constructor(private readonly chatbotService: ChatbotService) {}

  @Post('ask')
  async ask(
    @Body('message') message: string, 
    @Body('sessionId') sessionId: string
  ) {
    const id = sessionId || 'session-' + Date.now();
    return this.chatbotService.chat(message, id);
  }
}