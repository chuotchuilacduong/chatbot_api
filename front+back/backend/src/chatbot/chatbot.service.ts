import { Injectable, HttpException, HttpStatus } from '@nestjs/common';
import { HttpService } from '@nestjs/axios';
import { ConfigService } from '@nestjs/config';
import { firstValueFrom } from 'rxjs';

@Injectable()
export class ChatbotService {
  constructor(
    private readonly httpService: HttpService,
    private readonly configService: ConfigService,
  ) {}

  async chat(message: string, sessionId: string = 'default-session'): Promise<any> {
const apiUrl = this.configService.getOrThrow<string>('CHATBOT_API_URL');
    const payload = {
      query: message,      
      session_id: sessionId 
    };

    const headers = {
      'Content-Type': 'application/json',
    };

    try {
      const response = await firstValueFrom(
        this.httpService.post(apiUrl, payload, { headers }),
      );

      return {
        reply: response.data 
      };

    } catch (error) {
      console.error('Lỗi gọi API Python:', error.message);
      throw new HttpException(
        'Không thể kết nối tới máy chủ AI',
        HttpStatus.BAD_GATEWAY,
      );
    }
  }
}