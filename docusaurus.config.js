// @ts-check
// `@type` JSDoc annotations allow editor autocompletion and type checking
// (when paired with `@ts-check`).
// There are various equivalent ways to declare your Docusaurus config.
// See: https://docusaurus.io/docs/api/docusaurus-config

import {themes as prismThemes} from 'prism-react-renderer';
// 1. IMPORT THE MATH PLUGINS HERE:
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'LLM Training Handbook', // Updated to your book title
  tagline: 'A practical guide to training Large Language Models',
  favicon: 'img/favicon.ico',

  future: {
    v4: false,
  },

  url: 'https://hadikhd.github.io',
  baseUrl: '/llm-training-handbook/',

  organizationName: 'hadikhd',
  projectName: 'llm-traning-handbook',

  onBrokenLinks: 'throw',

  i18n: {
  defaultLocale: 'en',

  locales: ['en', 'fa'],

  localeConfigs: {
    en: {
      label: 'English',
      direction: 'ltr',
      htmlLang: 'en-US',
    },

    fa: {
      label: 'فارسی',
      direction: 'rtl',
      htmlLang: 'fa-IR',
    },
  },
},

  // ==========================================
  // MARKDOWN CONFIGURATION (FIXES ACORN ERRORS)
  // ==========================================
  markdown: {
    format: 'md', // Kept! This prevents the { } and < > errors in your book.
    // 'math: true' is REMOVED from here because it causes the TypeScript error.
  },

  // ==========================================
  // KATEX CSS STYLESHEET
  // ==========================================
  stylesheets: [
    {
      href: 'https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css',
      type: 'text/css',
      integrity: 'sha384-n8MVd4RsNIU0tAv4ct0nTaAbDJwPJzDEaqSD1odI+WdtXRGWt2kTvGFasHpSy3SV',
      crossorigin: 'anonymous',
    },
  ],

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: './sidebars.js',
          routeBasePath: '/',        // book = homepage
          
          // 2. ADD THE PLUGINS TO THE DOCS CONFIG HERE:
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],

          editUrl:
            'https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/',
        },
        blog: false, // Disabled for your book
        theme: {
          customCss: './src/css/custom.css',
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      image: 'img/docusaurus-social-card.jpg',
      colorMode: {
        respectPrefersColorScheme: true,
      },
      navbar: {
        title: 'LLM Training Handbook',
        logo: {
          alt: 'LLM Training Handbook Logo',
          src: 'img/logo.svg',
        },
        items: [
          {
            type: 'localeDropdown',
            position: 'left',
          },
          {
            type: 'docSidebar',
            sidebarId: 'tutorialSidebar',
            position: 'left',
            label: 'Chapters',
          },
          {
            href: 'https://github.com/facebook/docusaurus',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        copyright: `Copyright © ${new Date().getFullYear()} LLM Training Handbook, Inc. Built with Docusaurus.`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
      },
    }),
};

export default config;