import { Injectable, OnModuleInit } from '@nestjs/common';
import { PrismaClient } from '@prisma/client';

@Injectable()
export class PrismaService extends PrismaClient implements OnModuleInit {
  async onModuleInit() {
    // Connect to the database
    await this.$connect();
    console.log('📊 Connected to PostgreSQL database');
  }

  async onModuleDestroy() {
    // Disconnect from the database
    await this.$disconnect();
  }
}