from appium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import pandas as pd

# 打开app, 进入资料库
def enter_datafile(wait,driver):
    enter=wait.until(EC.presence_of_element_located((By.ID,'com.sankuai.moviepro:id/passport_policy_agree')))
    enter.click()
    print('已进入app，3秒后点击资料库')
    time.sleep(3)
    driver.tap([(757,510)],duration=500)
    print('已进入资料库')
    time.sleep(3)
    driver.tap([(722,736)],duration=500)
    print('动漫选择成功,请选择时间2017-2021！')
    time.sleep(10)
    print('进入资料库成功！')
    
# 爬取当前呈现在界面上的5个电影.
def crawl(wait,driver):
    x=147
    y=[698,1024,1308,1690,2013]
    revenue_xpath='/hierarchy/android.widget.FrameLayout/android.widget.LinearLayout/android.widget.FrameLayout/android.view.ViewGroup/android.widget.FrameLayout[2]/android.widget.FrameLayout/android.widget.ScrollView/android.widget.LinearLayout/android.widget.LinearLayout/android.widget.LinearLayout/android.widget.LinearLayout/android.widget.LinearLayout[1]/android.widget.LinearLayout[1]/android.widget.LinearLayout/android.widget.TextView[1]'
    measure_xpath='/hierarchy/android.widget.FrameLayout/android.widget.LinearLayout/android.widget.FrameLayout/android.view.ViewGroup/android.widget.FrameLayout[2]/android.widget.FrameLayout/android.widget.ScrollView/android.widget.LinearLayout/android.widget.LinearLayout/android.widget.LinearLayout/android.widget.LinearLayout/android.widget.LinearLayout[1]/android.widget.LinearLayout[1]/android.widget.LinearLayout/android.widget.TextView[2]'
    name_xpath='/hierarchy/android.widget.FrameLayout/android.widget.LinearLayout/android.widget.FrameLayout/android.view.ViewGroup/android.widget.FrameLayout[2]/android.widget.FrameLayout/android.widget.ScrollView/android.widget.LinearLayout/android.widget.LinearLayout/android.widget.RelativeLayout/android.widget.LinearLayout[2]/android.widget.TextView'
    drevenue_xpath='/hierarchy/android.widget.FrameLayout/android.widget.LinearLayout/android.widget.FrameLayout/android.view.ViewGroup/android.widget.FrameLayout[2]/android.widget.FrameLayout/android.widget.ScrollView/android.widget.LinearLayout/android.widget.LinearLayout/android.widget.LinearLayout/android.widget.LinearLayout/android.widget.LinearLayout[1]/android.widget.LinearLayout[3]/android.widget.LinearLayout/android.widget.TextView[1]'
    dmeasure_xpath='/hierarchy/android.widget.FrameLayout/android.widget.LinearLayout/android.widget.FrameLayout/android.view.ViewGroup/android.widget.FrameLayout[2]/android.widget.FrameLayout/android.widget.ScrollView/android.widget.LinearLayout/android.widget.LinearLayout/android.widget.LinearLayout/android.widget.LinearLayout/android.widget.LinearLayout[1]/android.widget.LinearLayout[3]/android.widget.LinearLayout/android.widget.TextView[2]'
    IPID='com.sankuai.moviepro:id/tv_type'
    timeid='com.sankuai.moviepro:id/tv_release_time_place'
    baiduindexpath='/hierarchy/android.widget.FrameLayout/android.widget.LinearLayout/android.widget.FrameLayout/android.view.ViewGroup/android.widget.FrameLayout[2]/android.widget.FrameLayout/android.widget.ScrollView/android.widget.LinearLayout/android.widget.LinearLayout[1]/android.widget.LinearLayout/android.widget.FrameLayout[3]/android.widget.LinearLayout/android.widget.LinearLayout[4]/android.widget.LinearLayout/android.widget.TextView'
    result={}
    for i in range(len(y)):
        time.sleep(3)
        driver.tap([(x,y[i])],duration=500)
        try:
            r=wait.until(EC.presence_of_element_located((By.XPATH,revenue_xpath)))
            revenue=r.text
        except:
            revenue='请手动查看'
        try:
            m=wait.until(EC.presence_of_element_located((By.XPATH,measure_xpath)))
            measure=m.text
        except:
            measure='请手动查看'
        try:
            n=wait.until(EC.presence_of_element_located((By.XPATH,name_xpath)))
            name=n.text
        except:
            name='请手动查看'
        try:
            dr=wait.until(EC.presence_of_element_located((By.XPATH,drevenue_xpath)))
            drevenue=dr.text
        except:
            drevenue='请手动查看'
        try:
            dm=wait.until(EC.presence_of_element_located((By.XPATH,dmeasure_xpath)))
            dmeasure=dm.text
        except:
            dmeasure='请手动查看'
        try:
            IP=wait.until(EC.presence_of_element_located((By.ID,IPID)))
        except:
            IP='请手动查看'
        try:
            Time=wait.until(EC.presence_of_element_located((By.ID,timeid)))
        except:
            Time='请手动查看'
        driver.swipe(659,2055,659,1746,5000)
        time.sleep(1)
        try:
            baiduindex=wait.until(EC.presence_of_element_located((By.XPATH,baiduindexpath)))   
            result[name]=[revenue+measure,drevenue+dmeasure,IP.text,Time.text,baiduindex.text]
        except:
            result[name]=[revenue+measure,drevenue+dmeasure,IP.text,Time.text,'暂无']
        print('获取成功！',result[name])
        driver.back()
    return result


# 打开app, 进入资料库后, 手动滑动屏幕, 每次依次爬取呈现在屏幕上的5个电影的资料
def main():
    server='http://localhost:4723/wd/hub'
    desired_caps={
        'platformName':'Android',
        'deviceName':'JEF_AN00',
        'appPackage':'com.sankuai.moviepro',
        'appActivity':'.views.activities.MainActivity'}
    driver=webdriver.Remote(server,desired_caps)
    wait=WebDriverWait(driver, 15)
    enter_datafile(wait, driver)
    backup={}
    for i in range(1,250,5):
        print('请在5秒内把{}号拖动至指定位置'.format(i))
        time.sleep(3)
        print('准备好了！')
        time.sleep(3)
        result=crawl(wait, driver)
        backup.update(result)
    index=['总票房','首日票房','影片分类','上映日期','热度指数']
    data=pd.DataFrame(backup,index=index)
    data=data.T
    data.to_excel('原始数据.xlsx')
    
    
main()
    

