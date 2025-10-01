import { Controller, Post, Body, UseGuards, Get, Req, Headers } from '@nestjs/common';
import { AuthService } from './auth.service';
import { RegisterDto, LoginDto } from './dto/auth.dto';
import { JwtAuthGuard } from './guards/jwt-auth.guard';

@Controller('auth')
export class AuthController {
  constructor(private authService: AuthService) {}

  @Post('register')
  async register(@Body() registerDto: RegisterDto) {
    return this.authService.register(registerDto);
  }

  @Post('login')
  async login(@Body() loginDto: LoginDto) {
    return this.authService.login(loginDto);
  }

  @Post('logout')
  @UseGuards(JwtAuthGuard)
  async logout(@Headers('authorization') authHeader: string) {
    const token = authHeader?.replace('Bearer ', '');
    return this.authService.logout(token);
  }

  @Get('me')
  @UseGuards(JwtAuthGuard)
  async getProfile(@Req() req) {
    return {
      user: req.user,
    };
  }

  @Get('validate')
  async validateToken(@Headers('authorization') authHeader: string) {
    const token = authHeader?.replace('Bearer ', '');
    
    if (!token) {
      return { valid: false };
    }

    const user = await this.authService.validateToken(token);
    return {
      valid: !!user,
      user: user || null,
    };
  }
}