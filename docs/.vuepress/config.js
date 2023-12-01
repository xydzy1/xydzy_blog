import { defaultTheme, defineUserConfig } from 'vuepress'

export default defineUserConfig({
  base: 'start',
  lang: 'zh-CN',
  title: 'xydzy的blog',
  description: 'xydzy的博客',
  theme: defaultTheme({
    navbar:[
        {text: '代码', link: '/code/'},
        {text: '数学', link: '/math/'},
        {text: '阅读', link: '/paper/'}
    ],
    sidebar: {
        '/code/': [
            'Intro',
            'git',
        ],
        '/math/': [
            'Intro',
        ],
        '/paper/': [
            'Intro',
        ]
    }

  })
})