package com.imooc

import org.apache.spark.ml.feature.StandardScaler
import org.apache.spark.ml.fpm.FPGrowth
import org.apache.spark.sql.SparkSession

object FPGrowth {
  def main(args: Array[String]): Unit = {


    val spark = SparkSession.builder()
      .master("local[*]")
      .getOrCreate()

    val sc = spark.sparkContext
    sc.setLogLevel("WARN")
    val data = sc.textFile("C:\\imooc\\transaction.txt")

    import spark.implicits._
    val transactions = data.map(x=>x.split(","))
      .toDF("item")

    val fpGrowth = new FPGrowth().setItemsCol("item")
      .setMinSupport(0.5) //最小支持度
      .setMinConfidence(0.6)

    val model = fpGrowth.fit(transactions)

    //展示频繁项集
    model.freqItemsets.show()



    /**
      *  fit -> run
      *
      *  genFreqItems
      *  第一次遍历
      *  过滤不符合最小支持度的item , 并排序
      *
      *  genFreqItemSets
      *  第二次遍历
      *  PFTress 操作
      *
      *  add      -> 向FPTress增加一条事务
      *  merge    -> 合并FPTress
      *  extract  -> 提取频繁项集
      *
      *
      */

  }

}
