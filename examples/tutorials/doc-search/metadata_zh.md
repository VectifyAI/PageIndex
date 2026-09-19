
## 按元数据搜索文档
<callout>支持元数据的 PageIndex 功能目前处于封闭测试阶段。请填写此表单以申请该功能的提前体验权限。</callout>

对于可以轻松通过元数据进行区分的文档，我们推荐使用元数据来搜索文档。
此方法特别适合以下文档类型：
- 按公司和时间段分类的财务报告
- 按案件类型分类的法律文件
- 按患者或病情分类的医疗记录
- 以及更多其他类型

在此类场景中，你可以利用文档的元数据进行搜索。一种流行的做法是使用 "Query to SQL" 来进行文档检索。

### 示例流程

#### 生成 PageIndex 树结构
将所有文档上传至 PageIndex，获取其 `doc_id`。

#### 建立 SQL 数据表

将文档连同其元数据和 PageIndex `doc_id` 一起存入数据库表中。

#### Query to SQL

使用 LLM 将用户的检索请求转换为 SQL 查询，以获取相关文档。

#### 使用 PageIndex 检索

使用已筛选出的文档的 PageIndex `doc_id`，通过 PageIndex 检索 API 进行进一步的检索。

## 💬 帮助与社区
如果你需要关于在你的场景中如何实施文档搜索的建议，请联系我们。

- 🤝 [加入我们的 Discord](https://discord.gg/VuXuf29EUj)  
- 📨 [给我们留言](https://ii2abc2jejf.typeform.com/to/meB40zV0)
