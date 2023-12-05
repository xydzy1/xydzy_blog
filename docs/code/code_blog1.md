# 实现vue点击目标区域之外触发事件

在dom挂载完成后添加事件，点击非目标区域触发事件

```html
<template>
    // stop用来阻止事件冒泡
    <div class="area_className" @click.stop="">

    </div>
<template>
```

```js
mounted(){
  // 模拟外部点击
  document.addEventListener('click', (e) => {
    this.function();
  })
},
// 组件生命周期结束时销毁监听事件
beforeDestroy() {
   window.removeEventListener('click', () => {}, true)
}
```
