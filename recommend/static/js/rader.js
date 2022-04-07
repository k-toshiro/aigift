// 各軸のラベル
var rlabels = ["ewc","cew","ecw","wecr","wcvr","rvr"];

// １つ目の系列の情報を設定
var series01name      = "ほしそうなもの";             // 系列１の名前
var series01data      = [1,2,3,4,5]; // 系列１データ
var series01bgcolor   = "rgba(255, 45, 132, 0.2)";            // 系列１の塗りつぶし色
var series01linecolor = "rgba(255, 45, 132, 1.0)";            // 系列１の線の色

// グラフ縦軸の最大／最小／目盛りの間隔を設定
var rmax  = 4;  // グラフ縦軸の最大
var rmin  = 0;  // グラフ縦軸の最小
var rstep = 1;  // グラフ縦軸の目盛り線を引く間隔

var ctx = document.getElementById("chart").getContext("2d");

var chart = new Chart(ctx, {
    type: "radar",
    data: {
        labels: rlabels,
        datasets:[
            {
            label:           series01name,
            data:            series01data,
            backgroundColor: series01bgcolor,
            borderColor:     series01linecolor,
            borderWidth:     1,
            }, 
        ]
    },
    options: {
        scales: {
            r: {
                display:      true,
                suggestedMax: rmax,
                suggestedMin: rmin,
                ticks: {
                stepSize: rstep,
                },
                grid: {
                    color: "white",
                  },
                pointLabels:{
                    color:"white",
                 },
                
            },
        },
    }
});