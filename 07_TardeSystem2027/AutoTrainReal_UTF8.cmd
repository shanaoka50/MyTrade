@echo off
rem コマンドプロンプトで使用する文字コードをUTF-8に設定する(NTFY対応)
chcp 65001 >nul

setlocal enabledelayedexpansion
rem TradingSystem2027_comandline.py のトレーニングを繰り返し行えるようにする

rem 仮想環境内のpython.exeへの絶対パスを指定します
set exec_prog_name=python
set exec_file_name=TradingSystem2027_comandline.py
set trade_system=System2027

set account_atrade_mode=REAL
set max_train_num=14
set train_num_each_batch=1

set symbol_start=0
set symbol_end=5

rem 通貨ペアを定義。配列のようにしているが、バッチファイルの場合は数字を入れ替えることができる
set symbol[0]=EURUSD
set symbol[1]=USDJPY
set symbol[2]=EURJPY
set symbol[3]=GBPJPY
set symbol[4]=GBPUSD
set symbol[5]=EURGBP
set symbol[6]=USDCHF
set symbol[7]=EURCHF
set symbol[8]=CHFJPY
set symbol[9]=GBPCHF
set symbol[10]=GBPAUD
set symbol[11]=GBPNZD

rem NTFYへ送るメッセージ
rem ntfyサーバーのURL（デフォルトは https://ntfy.sh ）
set NTFY_SERVER=https://ntfy.sh
rem 通知先トピック名（任意）
set NTFY_TOPIC=fxangler22_qawsedrftgyhujiko
rem メッセージ本文
set NTFY_MESSAGE=All Training Finished
rem タイトル
set NTFY_TITLE=%trade_system% %account_atrade_mode%
rem プライオリティ
set NTFY_PRIORITY=3
rem タグ(warning,loudspeaker等)
set NTFY_TAGS=loudspeaker

rem トレーニングのサブルーチン
:TRAIN_MODEL
echo トレーニングを開始します

rem 外側のforループは通貨ペア
FOR /L %%s IN (%symbol_start%,1,%symbol_end%) DO (
    echo symbol_num=%%s
    rem トレーニング実行
    echo %exec_prog_name% %exec_file_name% %account_atrade_mode% !symbol[%%s]! %max_train_num% %train_num_each_batch%
    rem 標準出力の最終行（トレーニング数）を取得するようにする
    for /f "delims=" %%L in ('%exec_prog_name% %exec_file_name% %account_atrade_mode% !symbol[%%s]! %max_train_num% %train_num_each_batch%') do (
		set "lastLine=%%L"
    )
    echo ERRORLEVEL=!ERRORLEVEL!
    if !ERRORLEVEL! equ 0 (
    	echo Pythonスクリプトは正常終了しました。
    ) else (
	echo Pythonスクリプトでエラーが発生しました。
	exit /b 1
    )
    set finish_train_num[%%s]=!lastLine!
    rem トレーニング終了数を取得する
    echo Symbol: !symbol[%%s]! Train has finished. TRANINGRESULT=!finish_train_num[%%s]!
)

rem すべての通貨ペアで最大トレーニング数が完了したかを判定する。終わっていなければサブルーチンを再処理する

rem 外側のforループは通貨ペア
FOR /L %%s IN (%symbol_start%,1,%symbol_end%) DO (
    rem 一つでも最大トレーニング数に達していなければトレーニングを継続する
    if !finish_train_num[%%s]! lss %max_train_num% (
        echo Symbol: !symbol[%%s]! FunishTrainNum !finish_train_num[%%s]! is less than %max_train_num% .
        goto TRAIN_MODEL
    )
)

rem すべてのトレーニングが終了したら、NTFYへメッセージを送る
rem ntfyへPOSTリクエストを送信
curl -H "Title: %NTFY_TITLE%" -H "Priority: %NTFY_PRIORITY%" -H "Tags: %NTFY_TAGS%" -d "%NTFY_MESSAGE%" %NTFY_SERVER%/%NTFY_TOPIC%

echo メッセージを送信しました。
