# README-dadafiles

这个文件专门来说明当前仓库中存在的各种数据文件. 

现在想移动这些文件已经不太可能了, 因为代码里充满了对这些文件的引用. 
不过可以将这些文件移动走, 并留下软链接. 

## 数据文件

GHCrawler
	- rawdata/ghs.csv: github 仓库信息, len(*) = 188039. 
      包含字段: 
      name,isFork,commits,branches,defaultBranch,releases,contributors,license
      watchers,stargazers,forks,size,createdAt,pushedAt,updatedAt,homepage,
      mainLanguage,totalIssues,openIssues,totalPullRequests,openPullRequests,
      lastCommit,lastCommitSHA,hasWiki,isArchived                             
    - export:
      - 看README.md文件即可. 这里绑架的数据有: 
        1. 用户 <=> 组织
        2. 用户 <=> 关注的人 (following)
        3. 用户 <=> 被关注的人 (followers)
        4. 用户 <=> watcher

        5. 仓库 <=> [] 标星的人
        6. 仓库 <=> [] 贡献者(人, 数量)

        7. 仓库 <=> 语言(若干, 包含它们的占比)
        8. 仓库 <=> 统计数据, 包含名称, 拥有者, 标星数量, 主题(topic); 
        9. 仓库 <=> [List] pr (#, user, 关闭?, merged?, 关联commit, reviewer)
        10.仓库 <=> [List] issue (#, committer, 状态, 内容)
        11.仓库 <=> pr, commit, [List] (文件名, patch, url, raw_url) 
            (patch数据并不全面, 需要使用url自行爬取)
        12.仓库 <=> README文件. (都在readme目录下)
        (这些文件中的数据都是以名称命名的, 还没有转换为编号)
