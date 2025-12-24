import { Module } from '@nestjs/common';
import { HttpModule } from '@nestjs/axios';
import { ChatbotService } from './chatbot.service';
import { ChatbotController } from './chatbot.controller';

@Module({
  imports: [HttpModule],
  controllers: [ChatbotController],
  providers: [ChatbotService],
})
export class ChatbotModule {}