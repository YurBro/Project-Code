%%  案例18: 基于Elman神经网络的电力负荷预测模型研究
% 
% 
% <html>
% <table border="0" width="600px" id="table1">	<tr>		<td><b><font size="2">该案例作者申明：</font></b></td>	</tr>	<tr>		<td><span class="comment"><font size="2">1：本人长期驻扎在此<a target="_blank" href="http://www.ilovematlab.cn/forum-158-1.html"><font color="#0000FF">板块</font></a>里，对<a target="_blank" href="http://www.ilovematlab.cn/thread-48362-1-1.html"><font color="#0000FF">该案例</font></a>提问，做到有问必答。</font></span></td></tr><tr>	<td><span class="comment"><font size="2">2：此案例有配套的教学视频，配套的完整可运行Matlab程序。</font></span></td>	</tr>	<tr>		<td><span class="comment"><font size="2">		3：以下内容为该案例的部分内容（约占该案例完整内容的1/10）。</font></span></td>	</tr>		<tr>		<td><span class="comment"><font size="2">		4：此案例为原创案例，转载请注明出处（<a target="_blank" href="http://www.ilovematlab.cn/">Matlab中文论坛</a>，<a target="_blank" href="http://www.ilovematlab.cn/forum-158-1.html">《Matlab神经网络30个案例分析》</a>）。</font></span></td>	</tr>		<tr>		<td><span class="comment"><font size="2">		5：若此案例碰巧与您的研究有关联，我们欢迎您提意见，要求等，我们考虑后可以加在案例里。</font></span></td>	</tr>		<tr>		<td><span class="comment"><font size="2">		6：您看到的以下内容为初稿，书籍的实际内容可能有少许出入，以书籍实际发行内容为准。</font></span></td>	</tr><tr>		<td><span class="comment"><font size="2">		7：此书其他常见问题、预定方式等，<a target="_blank" href="http://www.ilovematlab.cn/thread-47939-1-1.html">请点击这里</a>。</font></span></td>	</tr></table>
% </html>
% 


%% 清空环境变量

clc;
clear all
close all
nntwarn off;

%% 数据载入

load data;
a=data;

%% 选取训练数据和测试数据

for i=1:6
    p(i,:)=[a(i,:),a(i+1,:),a(i+2,:)];
end
% 训练数据输入
p_train=p(1:5,:);
% 训练数据输出
t_train=a(4:8,:);
% 测试数据输入
p_test=p(6,:);
% 测试数据输出
t_test=a(9,:);

% 为适应网络结构 做转置

p_train=p_train';
t_train=t_train';
p_test=p_test';


%% 网络的建立和训练
% 利用循环，设置不同的隐藏层神经元个数
nn=[7 11 14 18];
for i=1:4
    threshold=[0 1;0 1;0 1;0 1;0 1;0 1;0 1;0 1;0 1];
    % 建立Elman神经网络 隐藏层为nn(i)个神经元
    net=newelm(threshold,[nn(i),3],{'tansig','purelin'});
    % 设置网络训练参数
    net.trainparam.epochs=1000;
    net.trainparam.show=20;
    % 初始化网络
    net=init(net);
    % Elman网络训练
    net=train(net,p_train,t_train);
    % 预测数据
    y=sim(net,p_test);
    % 计算误差
    error(i,:)=y'-t_test;
end

%% 通过作图 观察不同隐藏层神经元个数时，网络的预测效果

plot(1:1:3,error(1,:),'-ro','linewidth',2);
hold on;
plot(1:1:3,error(2,:),'b:x','linewidth',2);
hold on;
plot(1:1:3,error(3,:),'k-.s','linewidth',2);
hold on;
plot(1:1:3,error(4,:),'c--d','linewidth',2);
title('Elman预测误差图')
set(gca,'Xtick',[1:3])
legend('7','11','14','18','location','best')
xlabel('时间点')
ylabel('误差')
hold off;

web browser http://www.ilovematlab.cn/viewthread.php?tid=63640
%%
%
% <html>
% <table align="center" >	<tr>		<td align="center"><font size="2">版权所有：</font><a
% href="http://www.ilovematlab.cn/">Matlab中文论坛</a>&nbsp;&nbsp; <script
% src="http://s3.cnzz.com/stat.php?id=971931&web_id=971931&show=pic" language="JavaScript" ></script>&nbsp;</td>	</tr></table>
% </html>
%