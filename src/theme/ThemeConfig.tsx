// @ts-check
// src/theme/ThemeConfig.tsx

import type { ThemeConfig } from '@docusaurus/theme-common';
import { themes as prismThemes } from 'prism-react-renderer';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

// این کد را در فایل ThemeConfig.tsx قرار دهید (جایگزین تمام کد قبلی بقیه تنظیمات themeConfig شود)

const themeConfig: ThemeConfig = {
  colorMode: {
    respectPrefersColorScheme: true,
  },

  // ==================== RTL با استثنای کامل کد و ریاضی ====================
  customCss: `
    /* ====================== RTL پایه + بادی محتوا ====================== */
    html[dir='rtl'],
    html[dir='rtl'] .theme-doc-root,
    html[dir='rtl'] .theme-doc-sidebar,
    html[dir='rtl'] .theme-doc-page,
    html[dir='rtl'] .theme-doc-sidebar-menu,
    html[dir='rtl'] .theme-doc-toc,
    html[dir='rtl'] article,
    html[dir='rtl'] .theme-doc-markdown,
    html[dir='rtl'] .rtl-content {
      direction: rtl;
      text-align: right;
      unicode-bidi: plaintext;
    }

    /* پاراگراف‌ها، لیست‌ها، blockquote */
    html[dir='rtl'] .theme-doc-markdown p,
    html[dir='rtl'] .theme-doc-markdown li,
    html[dir='rtl'] .theme-doc-markdown blockquote {
      direction: rtl;
      text-align: right;
    }

    /* ====================== استثنای کامل بلوک‌های فنی ====================== */
    /* کدهای بلاک (Python, JSON, Bash, JS, MDX, etc.) */
    html[dir='rtl'] .theme-doc-markdown pre,
    html[dir='rtl'] .theme-doc-markdown pre code,
    html[dir='rtl'] .theme-doc-markdown .codeBlockContainer,
    html[dir='rtl'] .theme-doc-markdown .prism-code,
    html[dir='rtl'] .theme-doc-markdown code[class*='language-'],
    html[dir='rtl'] .theme-doc-markdown .docusaurus-highlight-code-line {
      direction: ltr !important;
      text-align: left !important;
      unicode-bidi: isolate;
      white-space: pre !important;
    }

    /* inline code */
    html[dir='rtl'] .theme-doc-markdown :not(pre) > code {
      direction: ltr !important;
      unicode-bidi: isolate;
    }

    /* فرمول‌های ریاضی (KaTeX) */
    html[dir='rtl'] .theme-doc-markdown .katex,
    html[dir='rtl'] .theme-doc-markdown .katex-display,
    html[dir='rtl'] .theme-doc-markdown .katex-mathml {
      direction: ltr !important;
      text-align: center !important;
      unicode-bidi: isolate;
    }

    /* فرمت‌های جایگزین ریاضی */
    html[dir='rtl'] .theme-doc-markdown .MathJax,
    html[dir='rtl'] .theme-doc-markdown .mjx-container {
      direction: ltr !important;
      text-align: center !important;
      unicode-bidi: isolate;
    }

    /* جدول‌ها (اگر محتوا فنی دارند) */
    html[dir='rtl'] .theme-doc-markdown table {
      direction: rtl;
    }

    /* ====================== SIDEBAR و TOC (RTL اما کد داخلشان LTR) ====================== */
    html[dir='rtl'] .theme-doc-sidebar .menu__link,
    html[dir='rtl'] .theme-doc-sidebar .menu__link:hover,
    html[dir='rtl'] .theme-doc-toc__link {
      direction: rtl;
    }

    /* موبایل */
    @media (max-width: 996px) {
      html[dir='rtl'] .theme-doc-sidebar {
        direction: rtl;
      }
    }
  `,
};

// بقیه تنظیمات themeConfig را اینجا بیاورید (navbar، footer، prism و ...)

export default themeConfig;
