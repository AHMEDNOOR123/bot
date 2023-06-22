import pytholog as pl

## new knowledge base object
new_kb = pl.KnowledgeBase("flavor")
name = input("Enter name : ")
new_kb(["likes({x}, sausage)".format(x=name),
        "likes(melissa, pasta)",
        "likes(dmitry, cookie)",
        "likes(nikita, sausage)",
        "likes(assel, limonade)",
        "food_type(gouda, cheese)",
        "food_type(ritz, cracker)",
        "food_type(steak, meat)",
        "food_type(sausage, meat)",
        "food_type(limonade, juice)",
        "food_type(cookie, dessert)",
        "flavor(sweet, dessert)",
        "flavor(savory, meat)",
        "flavor(savory, cheese)",
        "flavor(sweet, juice)",
        "food_flavor(X, Y) :- food_type(X, Z), flavor(Y, Z)",
        "dish_to_like(X, Y) :- likes(X, L), food_type(L, T), flavor(F, T), food_flavor(Y, F), neq(L, Y)"])
# print(new_kb.query(pl.Expr("likes(noor, X)")))
y = new_kb.query(pl.Expr("likes({name}, X)".format(name=name)))[0]['X']
print(name+" likes "+y)