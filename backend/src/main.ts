import { NestFactory } from '@nestjs/core';
import { AppModule } from './app.module';

async function bootstrap() {
  const app = await NestFactory.create(AppModule);
  
  // Enable CORS for frontend communication
  app.enableCors({
    origin: ['http://localhost:3000'],
    credentials: true,
  });
  
  await app.listen(process.env.PORT ?? 8000);
  console.log('🚀 Backend server running on http://localhost:8000');
}
bootstrap();
