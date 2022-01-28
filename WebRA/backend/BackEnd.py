# -*- coding: utf-8 -*-
import json
from flask import Flask,request
import user_info_experiment as user_info

app = Flask(__name__)


# 浏览器向链接发送Post请求获取数据
@app.route('/getData',methods=['POST'])
def getData():
    print('调用get_data')
    responseDict = json.loads(str(request.get_data(), "utf8"))
    print(request.get_data())
    userid = responseDict.get('userid')

    # print("userid: "+str(userid))
    graph, option, inquiry_id, uid, title, text, head, tail, passdetermine = user_info.send_information(userid)
    # print(graph[0])
    # print(graph[1])
    # print(option)
    # print(inquiry_id)
    # print(graph[2])
    # print('graph[0]',graph[0])
    # print('graph[1]',graph[1])
    # print('graph[2]',graph[2])
    # print('options',option)
    # print('query_id',inquiry_id)
    # print('text',text)
    # print('title',title)
    # print(graph[3])
    return {
            'graph': {
                'nodes': graph[0],  # 节点 id 列表
                'edges': graph[1],
                'curve': graph[2],
                'hyperlinks': graph[3],
                'color':graph[4],
                # 'hyperlinks_r': graph[5]
            },
            'options': option,
            'queryid': inquiry_id,
            'text': text,
            'title':title,
            'head':head,
            'tail':tail,
            'passdetermine':passdetermine,
        }


# 浏览器将用户交互结果通过此链接Post至后端
@app.route('/returnResult',methods=['POST'])
def returnResult():
    print('调用 return_result')
    responseDict = json.loads(str(request.get_data(), "utf8"))
    print(request.get_data())
    result = responseDict.get('result')
    userid = responseDict.get('userid')
    queryid = responseDict.get('queryid')
    print("result: "+str(result))
    print("userid: "+str(userid))
    print("queryid: "+str(queryid))
    user_info.get_result(queryid, userid, result)
    return "OK"


# 开启服务
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # 0.0.0.0代表本机任何地址均可访问