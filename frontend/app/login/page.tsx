'use client';

import React, { useState } from 'react';
import { Container } from '@mui/material';
import LoginForm from '../../components/LoginForm';
import RegisterForm from '../../components/RegisterForm';
import ProtectedRoute from '../../components/ProtectedRoute';
import { useRouter } from 'next/navigation';

export default function LoginPage() {
  const [isLogin, setIsLogin] = useState(true);
  const router = useRouter();

  const handleSuccess = () => {
    router.push('/dashboard');
  };

  const handleSwitchMode = () => {
    setIsLogin(!isLogin);
  };

  return (
    <ProtectedRoute requireAuth={false}>
      <Container>
        {isLogin ? (
          <LoginForm 
            onSuccess={handleSuccess}
            onSwitchToRegister={handleSwitchMode}
          />
        ) : (
          <RegisterForm 
            onSuccess={handleSuccess}
            onSwitchToLogin={handleSwitchMode}
          />
        )}
      </Container>
    </ProtectedRoute>
  );
}