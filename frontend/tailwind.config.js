import plugin from 'tailwindcss/plugin'

/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx,ts,tsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['-apple-system', 'BlinkMacSystemFont', 'SF Pro Display', 'Segoe UI', 'sans-serif'],
      },
      colors: {
        // shadcn tokens (components/ui); values live in index.css as oklch components
        border: 'oklch(var(--border) / <alpha-value>)',
        muted: { DEFAULT: 'oklch(var(--muted) / <alpha-value>)', foreground: 'oklch(var(--muted-foreground) / <alpha-value>)' },
        primary: { DEFAULT: 'oklch(var(--primary) / <alpha-value>)', foreground: 'oklch(var(--primary-foreground) / <alpha-value>)' },
        ring: 'oklch(var(--ring) / <alpha-value>)',
        glass: {
          white: 'rgba(255,255,255,0.72)',
          border: 'rgba(255,255,255,0.45)',
        },
      },
      backdropBlur: {
        glass: '20px',
      },
      boxShadow: {
        glass: '0 8px 32px rgba(0,0,0,0.12), inset 0 1px 0 rgba(255,255,255,0.6)',
        'glass-lg': '0 20px 60px rgba(0,0,0,0.15), inset 0 1px 0 rgba(255,255,255,0.6)',
      },
    },
  },
  plugins: [
    // Tailwind 4's not-last: variant, used by the shadcn timeline
    plugin(({ addVariant }) => addVariant('not-last', '&:not(:last-child)')),
  ],
}
