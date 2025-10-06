/** @type {import('next').NextConfig} */
const nextConfig = {
  eslint: {
    ignoreDuringBuilds: true,
  },
  typescript: {
    ignoreBuildErrors: true,
  },
  // Enable turbopack for faster development
  turbo: {
    // Optimize turbopack settings
    root: '/Users/gracepatel/Documents/myProjects/Project1/frontend',
  },
  
  // Optimize for development
  reactStrictMode: true,
  swcMinify: true,
  
  // Reduce resource usage
  experimental: {
    // Reduce server memory usage
    serverComponentsExternalPackages: [],
    // Optimize builds
    optimizeCss: true,
    // Reduce bundle analyzer overhead
    bundlePagesExternals: false,
    // Optimize font loading
    optimizeFonts: true,
  },
  
  // HTTP configuration to prevent resource issues
  onDemandEntries: {
    // Period (in ms) where the server will keep pages in the buffer
    maxInactiveAge: 25 * 1000,
    // Number of pages that should be kept simultaneously without being disposed
    pagesBufferLength: 2,
  },
  
  // Compiler options
  compiler: {
    // Remove console.logs in production
    removeConsole: process.env.NODE_ENV === 'production',
  },
  
  // Webpack configuration to optimize resources
  webpack: (config, { dev, isServer }) => {
    // Optimize for development
    if (dev && !isServer) {
      config.optimization = {
        ...config.optimization,
        splitChunks: {
          chunks: 'all',
          cacheGroups: {
            vendor: {
              test: /[\\/]node_modules[\\/]/,
              name: 'vendors',
              chunks: 'all',
            },
          },
        },
      };
    }
    
    return config;
  },
  
  // Output settings
  output: 'standalone',
  
  // Environment variables
  env: {
    NEXT_TELEMETRY_DISABLED: '1',
  },
};

module.exports = nextConfig;