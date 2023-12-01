# 常用Git操作

## 远程仓库与本地的交互

### 连接远程仓库

连接

```sh
git remote add origin git@github.com:yourName/repository_name.git
git remote add origin https://github.com/yourName/repository_name.git
```

查看远程仓库地址

```sh
git remote -v
```

### 从远程仓库拉取文件

拉取远程仓库更新到本地

```sh
git pull origin "branch_name"
```

从远程仓库获取更新，但不自动合并

```sh
git remote update
```

### 本地分支连接远程分支

```sh
git branch -u origin/name
```

### 将更改上传到远程仓库

平时使用vscode的git插件实现即可。

```sh
git add <file1> <file2>
or 
git add <dir>
git commit -m "info"
git push origin/branch_name
```

上传更改到指定分支

```sh
git push --set-upstream origin [name] -u
```

### 撤销已经commit但未push的更改

```sh
git reset --soft HEAD~1
```

### 撤销已经push了的更改

```sh
git reset <commit id> <filename>
```

### 本地分支相关操作

新建本地分支

```sh
git checkout -b name
```

删除本地分支

```sh
git branch -d name
```

根据远程分支新建本地分支

```sh
git checkout -b name origin/name
```

### 切换分支时保留更改

暂存：

```sh
git stash
```

恢复：

```sh
git stash apply                //恢复最近一次暂存的修改(未从栈中删除)
git stash apply stash@{2}      //恢复索引 stash@{2} 对应的暂存的修改，索引可以通过 git stash list 进行查看
git stash drop stash@{1}        //删除 stash@{1} 分支对应的缓存数据
git stash pop                   //将最近一次暂存数据恢复并从栈中删除
```
