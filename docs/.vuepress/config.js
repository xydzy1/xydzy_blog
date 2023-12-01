import { defaultTheme } from 'vuepress'

export default {
  base: '/',
  lang: 'zh-CN',
  title: 'xydzy的blog',
  description: 'xydzy的博客',
  head: [
    [
      'link',{ rel: 'icon', href: 'blog_logo.jpg' }
    ]
  ],
  port: 3333,
  theme: defaultTheme({
    logo: 'blog_logo.jpg',
    navbar:[
        {text: '代码', link: '/code/'},
        {text: '数学', link: '/math/'},
        {text: '阅读', link: '/paper/'}
    ],
    sidebar: {
        '/code/': [
            'Intro',
            'git'
        ],
        '/math/': [
            'Intro'
        ],
        '/paper/': [
            'Intro'
        ]
    }
  })
}