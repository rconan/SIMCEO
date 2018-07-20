function getUrlVars() {
    var vars = {};
    var parts = window.location.href.replace(/[?&]+([^=&]+)=([^&]*)/gi,    
                                             function(m,key,value) {
                                                 vars[key] = value;
                                             });
    return vars;
}

AWS.config.region = 'us-west-2'; // Region
AWS.config.credentials = new AWS.CognitoIdentityCredentials({
    IdentityPoolId: 'us-west-2:b3b65d46-517b-46a9-b1dd-7c9c93be4f40',
});
var lambda = new AWS.Lambda();
var result = document.getElementById('result');
var urlvars = getUrlVars();
console.log(urlvars);
var action;
var input;

if (Object.keys(urlvars).length>0) {
    input = {
        AMI_ID: 'ami-6acc900a',
        instance_ID: urlvars['instance_ID'],
        region: 'us-west-1',
        action:  urlvars['action'],
        alpha: '3543'
    };
    console.log(input)
    result.innerHTML =  " EC2 instance "+urlvars['instance_ID']+" state: ";
    lambda.invoke({
        FunctionName: 'SIMCEO',
        Payload: JSON.stringify(input)
    }, function(err, data) {
        if (err) {
            console.log(err, err.stack);
            result.innerHTML = err;
        } else {
            var output = JSON.parse(data.Payload);
            result.innerHTML += output['status'];
            console.log(output)
        }
    });
}
