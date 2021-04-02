
const express = require('express')
const app = express()
const port = 3000

var spawn = require('child_process').spawn;

function getResultFromPython () {
    return new Promise((res,rej) => {

        var model = spawn('python', ['test.py' , 'param1']);

        model.stdout.on('data', function (data) {
            res(data.toString()); 
           
        });

        model.stderr.on('data', function (data) {
           rej(data.toString());
        });

    });
}


var bodyParser = require('body-parser')
app.use(bodyParser.json())

app.post('/', async (req, res) => {
    
    let result = await getResultFromPython(); 
    console.log(result); 
    res.send(result);

})



app.listen(port, () => {
    console.log(`Example app listening at http://localhost:${port}`)
})