#include <iostream>
#include <vector>
#include <utility>
#include <algorithm>
using namespace std;
typedef long long ll;
typedef long double ld;


vector<ll>
solution(ll a, ll b) {
    if (b == 0) {
        return {a, 1, 0};
    }
    vector<ll> res = solution(b, a % b);
    ll tmp = res[1];
    res[1] = res[2];
    res[2] = tmp - (a / b) * res[2];
    return res;
}

int main()
{
    ll a, b, c;
    cin >> a >> b >> c;
    vector<ll> ans = solution(a, b);
    if (c % ans[0] != 0) {
        cout << "No\n";
        return 0;
    }
    //ax + by = c
    // d = nod(a, b)
    // ax0 + by0 = c
    // x = b / d * n + x0;
    // y = a / d * n + y0;
    std::cout << ans[0] << " " << ans[1] << " " << ans[2] << "\n";
    ll d = ans[0];
    ll x = ans[1] * (c / d);
    ll y = ans[2] * (c / d);
    std::cout << d << " " << x << " " << y << "\n";

    while (x > 0) {
        if (b / d > 0) {
            x -= b / d;
            y += a / d;
        } else if (b / d < 0) {
            x += b / d;
            y -= a / d;
        } else {
            break;
        }
    }
    while (x <= 0) {
        if (b / d > 0) {
            x += b / d;
            y -= a / d;
        } else if (b / d < 0) {
            x -= b / d;
            y += a / d;
        } else {
            break;
        }
    }
    cout << x << " " << y << endl;
    return 0;
}

